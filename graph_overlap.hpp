/*
Copyright (c) 2015-2016 Xiaowei Zhu, Tsinghua University

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.
*/

#ifndef GRAPH_HPP
#define GRAPH_HPP

#include <stdio.h>
#include <cmath>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <fcntl.h>
#include <malloc.h>
#include <sys/mman.h>
#include <numa.h>
#include <omp.h>

#include <string>
#include <vector>
#include <thread>
#include <mutex>

#include "atomic.hpp"
#include "bitmap.hpp"
#include "constants.hpp"
#include "filesystem.hpp"
#include "mpi.hpp"
#include "time.hpp"
#include "type_overhead.hpp"
//#include "type.hpp"

enum ThreadStatus {
  WORKING,
  STEALING
};

enum MessageTag {
  ShuffleGraph,
  PassMessage,
  GatherVertexArray,
  Schedule
};

struct ThreadState {
  VertexId curr;
  VertexId end;
  ThreadStatus status;
};

struct MessageBuffer {
  size_t capacity;
  int count; // the actual size (i.e. bytes) should be sizeof(element) * count
  char * data;
  MessageBuffer () {
    capacity = 0;
    count = 0;
    data = NULL;
  }
  void init (int socket_id) {
    capacity = 4096;
    count = 0;
    data = (char*)numa_alloc_onnode(capacity, socket_id);
  }
  void resize(size_t new_capacity) {
    if (new_capacity > capacity) {
      char * new_data = (char*)numa_realloc(data, capacity, new_capacity);
      assert(new_data!=NULL);
      data = new_data;
      capacity = new_capacity;
    }
  }
};

template <typename MsgData>
struct MsgUnit {
  VertexId vertex;
  MsgData msg_data;
} __attribute__((packed));

template <typename EdgeData = Empty>
class Graph {
public:
  int partition_id,beg_id,end_id;
  int partitions;
  float over_rate;

  size_t alpha;

  int threads;
  int sockets;
  int threads_per_socket;

  size_t edge_data_size;
  size_t unit_size;
  size_t edge_unit_size;

  bool symmetric;
  VertexId vertices;
  VertexId org_vertices;

  EdgeId edges;
  VertexId * out_degree; // VertexId [vertices]; numa-aware
  VertexId * in_degree; // VertexId [vertices]; numa-aware

  VertexId * partition_offset; // VertexId [partitions+1]
  VertexId * new_partition_offset;
  VertexId * local_partition_offset; // VertexId [sockets+1]
  VertexId * partition_overlap_offset; // VertexId [partitions*2]
  VertexId * u_partition_overlap_offset;
  
  VertexId owned_vertices;
  EdgeId * outgoing_edges; // EdgeId [sockets]
  EdgeId * incoming_edges; // EdgeId [sockets]

  Bitmap ** incoming_adj_bitmap;
  EdgeId ** incoming_adj_index; // EdgeId [sockets] [vertices+1]; numa-aware
  AdjUnit<EdgeData> ** incoming_adj_list; // AdjUnit<EdgeData> [sockets] [vertices+1]; numa-aware
  Bitmap ** outgoing_adj_bitmap;
  EdgeId ** outgoing_adj_index; // EdgeId [sockets] [vertices+1]; numa-aware
  AdjUnit<EdgeData> ** outgoing_adj_list; // AdjUnit<EdgeData> [sockets] [vertices+1]; numa-aware

  VertexId * compressed_incoming_adj_vertices;
  CompressedAdjIndexUnit ** compressed_incoming_adj_index; // CompressedAdjIndexUnit [sockets] [...+1]; numa-aware
  VertexId * compressed_outgoing_adj_vertices;
  CompressedAdjIndexUnit ** compressed_outgoing_adj_index; // CompressedAdjIndexUnit [sockets] [...+1]; numa-aware

  ThreadState ** thread_state; // ThreadState* [threads]; numa-aware
  ThreadState ** tuned_chunks_dense; // ThreadState [partitions][threads];
  ThreadState ** tuned_chunks_sparse; // ThreadState [partitions][threads];
  VertexId *** tuned_chunks_currs; //VertexId [partitions][sockets][steps];
  VertexId *** tuned_chunks_ends; //VertexId [partitions][sockets][steps];
  VertexId ** stepsize;
  VertexId * steppointer;

  size_t local_send_buffer_limit;
  MessageBuffer ** local_send_buffer; // MessageBuffer* [threads]; numa-aware

  int current_send_part_id;
  MessageBuffer *** send_buffer; // MessageBuffer* [partitions] [sockets]; numa-aware
  MessageBuffer *** recv_buffer; // MessageBuffer* [partitions] [sockets]; numa-aware

  unsigned long long * comm_time; // precise timer
  unsigned long long ** timekeeper;
  unsigned long long * compute_time;

  VertexHead vertex_head;
  SubsetLink subset_head;

  Graph() {
    threads = numa_num_configured_cpus();
    sockets = numa_num_configured_nodes();
    threads_per_socket = threads / sockets;

    init();
  }

  inline int get_socket_id(int thread_id) {
    return thread_id / threads_per_socket;
  }

  inline int get_socket_offset(int thread_id) {
    return thread_id % threads_per_socket;
  }

  void init() {
    over_rate = 0.2;
    edge_data_size = std::is_same<EdgeData, Empty>::value ? 0 : sizeof(EdgeData);
    unit_size = sizeof(VertexId) + edge_data_size;
    edge_unit_size = sizeof(VertexId) + unit_size;

    assert( numa_available() != -1 );
    assert( sizeof(unsigned long) == 8 ); // assume unsigned long is 64-bit

    char nodestring[sockets*2+1];
    nodestring[0] = '0';
    for (int s_i=1;s_i<sockets;s_i++) {
      nodestring[s_i*2-1] = ',';
      nodestring[s_i*2] = '0'+s_i;
    }
    struct bitmask * nodemask = numa_parse_nodestring(nodestring);
    numa_set_interleave_mask(nodemask);

    subset_head = NULL;

    omp_set_dynamic(0);
    omp_set_num_threads(threads);
    thread_state = new ThreadState * [threads];
    local_send_buffer_limit = 16;
    local_send_buffer = new MessageBuffer * [threads];
    for (int t_i=0;t_i<threads;t_i++) {
      thread_state[t_i] = (ThreadState*)numa_alloc_onnode( sizeof(ThreadState), get_socket_id(t_i));
      local_send_buffer[t_i] = (MessageBuffer*)numa_alloc_onnode( sizeof(MessageBuffer), get_socket_id(t_i));
      local_send_buffer[t_i]->init(get_socket_id(t_i));
    }
    #pragma omp parallel for
    for (int t_i=0;t_i<threads;t_i++) {
      int s_i = get_socket_id(t_i);
      assert(numa_run_on_node(s_i)==0);
      #ifdef PRINT_DEBUG_MESSAGES
      // printf("thread-%d bound to socket-%d\n", t_i, s_i);
        //omp, distribute threads, share-memory
      #endif
    }
    #ifdef PRINT_DEBUG_MESSAGES
    // printf("threads=%d*%d\n", sockets, threads_per_socket);
    // printf("interleave on %s\n", nodestring);
    #endif

    MPI_Comm_rank(MPI_COMM_WORLD, &partition_id);
    MPI_Comm_size(MPI_COMM_WORLD, &partitions);
    if(partitions==1){
      beg_id = 0;
      end_id = 1;
    }else{
      beg_id = partition_id*2;
      end_id = (partition_id*2+3)%(partitions*2);
    }
    #ifdef PRINT_DEBUG_MESSAGES
    printf("partition %d: beg_id %d,end_id %d\n",partition_id,beg_id,end_id);
    #endif

    comm_time = new unsigned long long [partitions];
    timekeeper = new unsigned long long * [partitions];
    compute_time = new unsigned long long [partitions];
    for(int i=0;i<partitions;i++)
      timekeeper[i]=new unsigned long long [partitions];

    send_buffer = new MessageBuffer ** [partitions];
    recv_buffer = new MessageBuffer ** [partitions];
    for (int i=0;i<partitions;i++) {
      send_buffer[i] = new MessageBuffer * [sockets];
      recv_buffer[i] = new MessageBuffer * [sockets];
      for (int s_i=0;s_i<sockets;s_i++) {
        send_buffer[i][s_i] = (MessageBuffer*)numa_alloc_onnode( sizeof(MessageBuffer), s_i);
        send_buffer[i][s_i]->init(s_i);
        recv_buffer[i][s_i] = (MessageBuffer*)numa_alloc_onnode( sizeof(MessageBuffer), s_i);
        recv_buffer[i][s_i]->init(s_i);
      }
    }

    //alpha = 8 * (partitions - 1);
    alpha = 0;

    MPI_Barrier(MPI_COMM_WORLD);
  }

  // fill a vertex array with a specific value
  template<typename T>
  void fill_vertex_array(T * array, T value) {
    #pragma omp parallel for
    for (VertexId v_i=partition_overlap_offset[beg_id];v_i<partition_overlap_offset[end_id];v_i++) {
      if(v_i<0)
        array[v_i + vertices] = value;
      else if(v_i>=vertices)
        array[v_i - vertices] = value;
      else
        array[v_i] = value;
    }
  }

  // allocate a numa-aware vertex array
  template<typename T>
  T * alloc_vertex_array() {
    char * array = (char *)mmap(NULL, sizeof(T) * vertices, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    //edit by wyd
    VertexArray * vertex_array_unit = new VertexArray(array,sizeof(T));
    if(vertex_head.headaddr==NULL)
      vertex_head.headaddr = vertex_array_unit;
    else{
      VertexArray * p = vertex_head.headaddr;
      while(p->next!=NULL)
        p = p->next;
      p->next = vertex_array_unit;
    }
    //edit by wyd

    assert(array!=NULL);
    for (int s_i=0;s_i<sockets;s_i++) {
      if(local_partition_offset[s_i]>=0 && local_partition_offset[s_i+1]<=vertices)
        numa_tonode_memory(array + sizeof(T) * local_partition_offset[s_i], sizeof(T) * (local_partition_offset[s_i+1] - local_partition_offset[s_i]), s_i);
      else if(local_partition_offset[s_i]<0){
        if(local_partition_offset[s_i+1]<0)
          numa_tonode_memory(array + sizeof(T) * (local_partition_offset[s_i] + vertices) % vertices, sizeof(T) * (local_partition_offset[s_i+1] - local_partition_offset[s_i]), s_i);
        else{
          numa_tonode_memory(array + sizeof(T) * (local_partition_offset[s_i] + vertices) % vertices, sizeof(T) * (0 - local_partition_offset[s_i]), s_i);
          numa_tonode_memory(array, sizeof(T) * local_partition_offset[s_i+1], s_i);
        }
      }
      else{
        if(local_partition_offset[s_i]>=vertices)
          numa_tonode_memory(array + sizeof(T) * local_partition_offset[s_i] % vertices, sizeof(T) * (local_partition_offset[s_i+1] - local_partition_offset[s_i]), s_i);
        else{
          numa_tonode_memory(array + sizeof(T) * local_partition_offset[s_i], sizeof(T) * (vertices - local_partition_offset[s_i]), s_i);
          numa_tonode_memory(array, sizeof(T) * (local_partition_offset[s_i+1] % vertices), s_i);
        }
      }
    }
    return (T*)array;
  }

  // deallocate a vertex array
  template<typename T>
  int dealloc_vertex_array(T * array) {
    VertexArray * r = vertex_head.headaddr;
    VertexArray * p = r->next;
    if(reinterpret_cast<T*>(r->add) == array){
      free(r);
    }else{
      while(reinterpret_cast<T*>(p->add) != array){
        r=p;
        p=p->next;
      }
      r->next=p->next;
      free(p);
    }
    numa_free(array, sizeof(T) * vertices);
  }

  // allocate a numa-oblivious vertex array
  template<typename T>
  T * alloc_interleaved_vertex_array() {
    T * array = (T *)numa_alloc_interleaved( sizeof(T) * vertices );
    assert(array!=NULL);
    return array;
  }

  // dump a vertex array to path
  template<typename T>
  void dump_vertex_array(T * array, std::string path) {
    long file_length = sizeof(T) * vertices;
    if (!file_exists(path) || file_size(path) != file_length) {
      if (partition_id==0) {
        FILE * fout = fopen(path.c_str(), "wb");
        char * buffer = new char [PAGESIZE];
        for (long offset=0;offset<file_length;) {
          if (file_length - offset >= PAGESIZE) {
            fwrite(buffer, 1, PAGESIZE, fout);
            offset += PAGESIZE;
          } else {
            fwrite(buffer, 1, file_length - offset, fout);
            offset += file_length - offset;
          }
        }
        fclose(fout);
      }
      MPI_Barrier(MPI_COMM_WORLD);
    }
    int fd = open(path.c_str(), O_RDWR);
    assert(fd!=-1);
    if(partition_offset[partition_id]>=0 && partition_offset[partition_id+1]<=vertices){
      long offset = sizeof(T) * partition_offset[partition_id];
      long end_offset = sizeof(T) * partition_offset[partition_id+1];
      void * data = (void *)array;
      assert(lseek(fd, offset, SEEK_SET)!=-1);
      while (offset < end_offset) {
        long bytes = write(fd, data + offset, end_offset - offset);
        assert(bytes!=-1);
        offset += bytes;
      }
    }
    else if(partition_offset[partition_id]<0){
      long offset = sizeof(T) * (vertices+ partition_offset[partition_id]);
      long end_offset = sizeof(T) * partition_offset[partition_id+1];
      void * data = (void *)array;
      assert(lseek(fd, offset, SEEK_SET)!=-1);
      write(fd, data + offset, sizeof(T) * vertices - offset);
      assert(lseek(fd, 0, SEEK_SET)!=-1);
      write(fd, data, end_offset);
    }
    else if(partition_offset[partition_id+1]>vertices){
      long offset = sizeof(T) * partition_offset[partition_id];
      long end_offset = sizeof(T) * (partition_offset[partition_id+1]-vertices);
      void * data = (void *)array;
      assert(lseek(fd, offset, SEEK_SET)!=-1);
      write(fd, data + offset, sizeof(T) * vertices - offset);
      assert(lseek(fd, 0, SEEK_SET)!=-1);
      write(fd, data, end_offset);
    }
    assert(close(fd)==0);
  }

  void dump_vertex_subset(unsigned long  * array, std::string path) {
    long file_length = WORD_OFFSET(vertices);
    if (!file_exists(path) || file_size(path) != file_length) {
      if (partition_id==0) {
        FILE * fout = fopen(path.c_str(), "wb");
        char * buffer = new char [PAGESIZE];
        for (long offset=0;offset<file_length;) {
          if (file_length - offset >= PAGESIZE) {
            fwrite(buffer, 1, PAGESIZE, fout);
            offset += PAGESIZE;
          } else {
            fwrite(buffer, 1, file_length - offset, fout);
            offset += file_length - offset;
          }
        }
        fclose(fout);
      }
      MPI_Barrier(MPI_COMM_WORLD);
    }
    int fd = open(path.c_str(), O_RDWR);
    assert(fd!=-1);
    if(partition_offset[partition_id]>=0 && partition_offset[partition_id+1]<=vertices){
      long offset = WORD_OFFSET(partition_offset[partition_id]);
      long end_offset = WORD_OFFSET(partition_offset[partition_id+1]);
      void * data = (void *)array;
      assert(lseek(fd, offset, SEEK_SET)!=-1);
      while (offset < end_offset) {
        long bytes = write(fd, data + offset, end_offset - offset);
        assert(bytes!=-1);
        offset += bytes;
      }
    }
    else if(partition_offset[partition_id]<0){
      long offset = WORD_OFFSET(vertices+ partition_offset[partition_id]);
      long end_offset = WORD_OFFSET(partition_offset[partition_id+1]);
      void * data = (void *)array;
      assert(lseek(fd, offset, SEEK_SET)!=-1);
      write(fd, data + offset, WORD_OFFSET(vertices) - offset);
      assert(lseek(fd, 0, SEEK_SET)!=-1);
      write(fd, data, end_offset);
    }
    else if(partition_offset[partition_id+1]>vertices){
      long offset = WORD_OFFSET(partition_offset[partition_id]);
      long end_offset = WORD_OFFSET(partition_offset[partition_id+1]-vertices);
      void * data = (void *)array;
      assert(lseek(fd, offset, SEEK_SET)!=-1);
      write(fd, data + offset, WORD_OFFSET(vertices) - offset);
      assert(lseek(fd, 0, SEEK_SET)!=-1);
      write(fd, data, end_offset);
    }
    assert(close(fd)==0);
  }
  // restore a vertex array from path
  template<typename T>
  void restore_vertex_array(T * array, std::string path) {
    long file_length = sizeof(T) * vertices;
    if (!file_exists(path) || file_size(path) != file_length) {
      assert(false);
    }
    int fd = open(path.c_str(), O_RDWR);
    assert(fd!=-1);
    long offset = sizeof(T) * partition_offset[partition_id];
    long end_offset = sizeof(T) * partition_offset[partition_id+1];
    void * data = (void *)array;
    assert(lseek(fd, offset, SEEK_SET)!=-1);
    while (offset < end_offset) {
      long bytes = read(fd, data + offset, end_offset - offset);
      assert(bytes!=-1);
      offset += bytes;
    }
    assert(close(fd)==0);
  }

  // gather a vertex array
  template<typename T>
  void gather_vertex_array(T * array, int root) {  
    if (partition_id!=root) {
      if(partition_offset[partition_id]>=0 && partition_offset[partition_id+1]<=vertices)
        MPI_Send(array + partition_offset[partition_id], sizeof(T) * (partition_offset[partition_id+1]-partition_offset[partition_id]), MPI_CHAR, root, GatherVertexArray, MPI_COMM_WORLD);
      else if(partition_offset[partition_id]<0){
        MPI_Send(array + (vertices + partition_offset[partition_id]), sizeof(T) * (0-partition_offset[partition_id]), MPI_CHAR, root, GatherVertexArray, MPI_COMM_WORLD);
        MPI_Send(array, sizeof(T) * partition_offset[partition_id+1], MPI_CHAR, root, GatherVertexArray, MPI_COMM_WORLD);
      }
      else{
        MPI_Send(array + partition_offset[partition_id], sizeof(T) * (vertices-partition_offset[partition_id]), MPI_CHAR, root, GatherVertexArray, MPI_COMM_WORLD);
        MPI_Send(array, sizeof(T) * partition_offset[partition_id+1]%vertices, MPI_CHAR, root, GatherVertexArray, MPI_COMM_WORLD);
      }
    } else {
      for (int i=0;i<partitions;i++) {
        if (i==partition_id) continue;
        MPI_Status recv_status;
        if(partition_offset[i]>=0 && partition_offset[i+1]<=vertices)
          MPI_Recv(array + partition_offset[i], sizeof(T) * (partition_offset[i + 1] - partition_offset[i]), MPI_CHAR, i, GatherVertexArray, MPI_COMM_WORLD, &recv_status);
        else if(partition_offset[i]<0){
          MPI_Recv(array + (vertices + partition_offset[i]), sizeof(T) * (0-partition_offset[i]), MPI_CHAR, i, GatherVertexArray, MPI_COMM_WORLD, &recv_status);
          MPI_Recv(array,sizeof(T) * partition_offset[i+1], MPI_CHAR, i, GatherVertexArray, MPI_COMM_WORLD, &recv_status);
        }
        else{
          MPI_Recv(array + partition_offset[i], sizeof(T) * (vertices-partition_offset[i]), MPI_CHAR, i, GatherVertexArray, MPI_COMM_WORLD, &recv_status);
          MPI_Recv(array,sizeof(T) * partition_offset[i+1]%vertices, MPI_CHAR, i, GatherVertexArray, MPI_COMM_WORLD, &recv_status);
        }
      }
    }
  }

  // allocate a vertex subset
  VertexSubset * alloc_vertex_subset() {
    SubsetLink r = (SubsetLink)malloc(sizeof(SubsetUnit));
    r = (SubsetLink)malloc(sizeof(SubsetUnit));
    r->a = new VertexSubset(vertices);
    r->next = NULL;
    if(subset_head==NULL){
      subset_head = r;
    }
    else{
      SubsetLink p = subset_head;
      while(p->next!=NULL)
        p = p->next;
      p->next = r;
    }
    return r->a;
  }
  // deallocate a vertex subset
  template<typename T>
  int dealloc_vertex_subset(T * array){
    SubsetLink r = subset_head;
    SubsetLink p = r->next;
    if(reinterpret_cast<T*>(r->a) == array){
      free(r);
      subset_head = NULL;
    }else{
      while(reinterpret_cast<T*>(p->a) != array){
        r=p;
        p=p->next;
      }
      r->next=p->next;
      free(p);
    }
    delete array;
  }

  int get_partition_id(VertexId v_i){
    for (int i=0;i<partitions;i++) {
      if (v_i >= partition_offset[i] && v_i < partition_offset[i+1]) {
        return i;
      }
    }
    assert(false);
  }

  int get_local_partition_id(VertexId v_i){
    for (int s_i=0;s_i<sockets;s_i++) {
      if (v_i >= local_partition_offset[s_i] && v_i < local_partition_offset[s_i+1]) {
        return s_i;
      }
    }
    v_i = v_i - vertices;
    for (int s_i=0;s_i<sockets;s_i++) {
      if (v_i >= local_partition_offset[s_i] && v_i < local_partition_offset[s_i+1]) {
        return s_i;
      }
    }
    v_i = v_i + vertices + vertices;
    for (int s_i=0;s_i<sockets;s_i++) {
      if (v_i >= local_partition_offset[s_i] && v_i < local_partition_offset[s_i+1]) {
        return s_i;
      }
    }
    fprintf(stderr,"404 found %d at %d, %d %d %d,vtxs %d\n",v_i-vertices,partition_id,local_partition_offset[0],local_partition_offset[1],local_partition_offset[2],vertices);
    assert(false);
  }

  int get_overlap_id(VertexId v_i){
    for (int i=1;i<partitions*2-1;i++) {
      if (v_i >= u_partition_overlap_offset[i] && v_i < u_partition_overlap_offset[i+1]) {
        return i;
      }
    }
    if (v_i >= u_partition_overlap_offset[partitions*2-1] && v_i < u_partition_overlap_offset[0])
      return (partitions*2-1);
    return 0;
  }
  // load a directed graph and make it undirected
  void load_undirected_from_directed(std::string path, VertexId vertices) {
    double prep_time = 0;
    prep_time -= MPI_Wtime();

    symmetric = true;

    MPI_Datatype vid_t = get_mpi_data_type<VertexId>();

    vertices = (vertices/PAGESIZE + 1) * PAGESIZE;
    this->vertices = vertices;
    long total_bytes = file_size(path.c_str());
    this->edges = total_bytes / edge_unit_size;
    #ifdef PRINT_DEBUG_MESSAGES
    if (partition_id==0) {
      printf("|V| = %u, |E| = %lu\n", vertices, edges);
    }
    #endif

    EdgeId read_edges = edges / partitions;
    if (partition_id==partitions-1) {
      read_edges += edges % partitions;
    }
    long bytes_to_read = edge_unit_size * read_edges;
    long read_offset = edge_unit_size * (edges / partitions * partition_id);
    long read_bytes;
    int fin = open(path.c_str(), O_RDONLY);
    EdgeUnit<EdgeData> * read_edge_buffer = new EdgeUnit<EdgeData> [CHUNKSIZE];

    out_degree = alloc_interleaved_vertex_array<VertexId>();
    for (VertexId v_i=0;v_i<vertices;v_i++) {
      out_degree[v_i] = 0;
    }
    assert(lseek(fin, read_offset, SEEK_SET)==read_offset);
    read_bytes = 0;
    while (read_bytes < bytes_to_read) {
      long curr_read_bytes;
      if (bytes_to_read - read_bytes > edge_unit_size * CHUNKSIZE) {
        curr_read_bytes = read(fin, read_edge_buffer, edge_unit_size * CHUNKSIZE);
      } else {
        curr_read_bytes = read(fin, read_edge_buffer, bytes_to_read - read_bytes);
      }
      assert(curr_read_bytes>=0);
      read_bytes += curr_read_bytes;
      EdgeId curr_read_edges = curr_read_bytes / edge_unit_size;
      // #pragma omp parallel for
      for (EdgeId e_i=0;e_i<curr_read_edges;e_i++) {
        VertexId src = read_edge_buffer[e_i].src;
        VertexId dst = read_edge_buffer[e_i].dst;
        __sync_fetch_and_add(&out_degree[src], 1);
        __sync_fetch_and_add(&out_degree[dst], 1);
      }
    }
    MPI_Allreduce(MPI_IN_PLACE, out_degree, vertices, vid_t, MPI_SUM, MPI_COMM_WORLD);

    // locality-aware chunking
    partition_offset = new VertexId [partitions + 1];
    new_partition_offset = new VertexId [partitions + 1];
    partition_overlap_offset = new VertexId [partitions * 2];
    partition_offset[0] = 0;
    partition_overlap_offset[0] = 0;
    EdgeId remained_amount = edges * 2 + EdgeId(vertices) * alpha;
    for (int i=0;i<partitions;i++){
      VertexId remained_partitions = partitions - i;
      EdgeId expected_chunk_size = remained_amount / remained_partitions;
      EdgeId expected_overlap_size = expected_chunk_size * over_rate;
      if (remained_partitions==1) {
        partition_offset[i+1] = vertices;
        EdgeId got_edges = 0;
        bool beg_tag=true;
        for (VertexId v_i=partition_offset[i];v_i<vertices;v_i++) {
          got_edges += out_degree[v_i] + alpha;
          if (beg_tag && got_edges > expected_overlap_size){
            partition_overlap_offset[2*i+1] = v_i;
            beg_tag = false;
          } else if (got_edges > expected_chunk_size - expected_overlap_size){
            partition_overlap_offset[0]=v_i;
            break;
          }
        }
        partition_overlap_offset[2*i+1] = (partition_overlap_offset[2*i+1]) / PAGESIZE * PAGESIZE;
        partition_overlap_offset[0] = (vertices - partition_overlap_offset[0]) / PAGESIZE * PAGESIZE;
        partition_overlap_offset[0] = 0 - partition_overlap_offset[0];
        u_partition_overlap_offset[2*i+1] = partition_overlap_offset[2*i+1];
        u_partition_overlap_offset[0] = partition_overlap_offset[0] + vertices;
      } else {
        EdgeId got_edges = 0;
        bool beg_tag=true,mid_tag=true;
        for (VertexId v_i=partition_offset[i];v_i<vertices;v_i++) {
          got_edges += out_degree[v_i] + alpha;
          if (beg_tag && got_edges > expected_overlap_size){
            partition_overlap_offset[2*i+1] = v_i;
            beg_tag = false;
          }
          else if (mid_tag && got_edges > expected_chunk_size - expected_overlap_size){
            partition_overlap_offset[2*i+2] = v_i;
            mid_tag = false;
          }
          else if (got_edges > expected_chunk_size){
            partition_offset[i+1] = v_i;
            break;
          }
        }
        partition_offset[i+1] = (partition_offset[i+1]) / PAGESIZE * PAGESIZE; // aligned with pages
        partition_overlap_offset[2*i+1] = (partition_overlap_offset[2*i+1]) / PAGESIZE * PAGESIZE;
        partition_overlap_offset[2*i+2] = (partition_overlap_offset[2*i+2]) / PAGESIZE * PAGESIZE;
        if(partition_overlap_offset[2*i+2]<partition_overlap_offset[2*i+1]){
          VertexId tmp = partition_overlap_offset[2*i+1];
          partition_overlap_offset[2*i+1] = partition_overlap_offset[2*i+2];
          partition_overlap_offset[2*i+2] = tmp;
        }
        u_partition_overlap_offset[2*i+1] = partition_overlap_offset[2*i+1];
        u_partition_overlap_offset[2*i+2] = partition_overlap_offset[2*i+2];
      }
      for (VertexId v_i=partition_offset[i];v_i<partition_offset[i+1];v_i++) {
        remained_amount -= out_degree[v_i] + alpha;
      }
    }
    partition_overlap_offset[1]=(vertices + partition_overlap_offset[1] - partition_overlap_offset[partitions*2-2]) / PAGESIZE * PAGESIZE +  partition_overlap_offset[partitions*2-2];
    u_partition_overlap_offset[1] = partition_overlap_offset[1] - vertices;
    assert(partition_offset[partitions]==vertices);
    owned_vertices = partition_overlap_offset[end_id] - partition_overlap_offset[beg_id]; 

    // check consistency of partition boundaries
    VertexId * global_partition_offset = new VertexId [partitions + 1];
    MPI_Allreduce(partition_offset, global_partition_offset, partitions + 1, vid_t, MPI_MAX, MPI_COMM_WORLD);
    for (int i=0;i<=partitions;i++) {
      assert(partition_offset[i] == global_partition_offset[i]);
    }
    MPI_Allreduce(partition_offset, global_partition_offset, partitions + 1, vid_t, MPI_MIN, MPI_COMM_WORLD);
    for (int i=0;i<=partitions;i++) {
      assert(partition_offset[i] == global_partition_offset[i]);
    }
    #ifdef PRINT_DEBUG_MESSAGES
    if (partition_id==0) {
      for (int i=0;i<partitions;i++) {
        EdgeId part_out_edges = 0;
        for (VertexId v_i=partition_offset[i];v_i<partition_offset[i+1];v_i++) {
          part_out_edges += out_degree[v_i];
        }
        printf("|V'_%d| = %d |E_%d| = %d\n", i, partition_offset[i+1] - partition_offset[i], i, part_out_edges);
        printf("overlap[%d] = %d,overlap[%d] = %d\n",2*i,partition_overlap_offset[2*i],2*i+1,partition_overlap_offset[2*i+1]);
      }
    }
    MPI_Barrier(MPI_COMM_WORLD);
    #endif
    delete [] global_partition_offset;
    {
      // NUMA-aware sub-chunking
      local_partition_offset = new VertexId [sockets + 1];
      EdgeId part_out_edges = 0;
      for (VertexId v_i=partition_overlap_offset[beg_id];v_i<partition_overlap_offset[end_id];v_i++) {
        VertexId v_mod = v_i;
        if(v_i<0) v_mod = v_i + vertices;
        else if(v_i>=vertices) v_mod = v_i - vertices;
        part_out_edges += out_degree[v_mod];
        // partition out_edges
      }
      local_partition_offset[0] = partition_overlap_offset[beg_id];
      EdgeId remained_amount = part_out_edges + EdgeId(owned_vertices) * alpha;
      for (int s_i=0;s_i<sockets;s_i++) {
        VertexId remained_partitions = sockets - s_i;
        EdgeId expected_chunk_size = remained_amount / remained_partitions;
        if (remained_partitions==1) {
          local_partition_offset[s_i+1] = partition_overlap_offset[end_id];
        } else {
          EdgeId got_edges = 0;
          for (VertexId v_i=local_partition_offset[s_i];v_i<partition_overlap_offset[end_id];v_i++) {
            VertexId v_mod = v_i;
            if (v_i<0) v_mod = v_i + vertices;
            else if (v_i>=vertices) v_mod = v_i - vertices;
            got_edges += out_degree[v_mod] + alpha;
            if (got_edges > expected_chunk_size) {
              local_partition_offset[s_i+1] = v_i;
              break;
            }
          }
          local_partition_offset[s_i+1] = (local_partition_offset[s_i+1]) / PAGESIZE * PAGESIZE; // aligned with pages
        }
        EdgeId sub_part_out_edges = 0;
        for (VertexId v_i=local_partition_offset[s_i];v_i<local_partition_offset[s_i+1];v_i++) {
          remained_amount -= out_degree[v_i] + alpha;
          sub_part_out_edges += out_degree[v_i];
        }
        #ifdef PRINT_DEBUG_MESSAGES
        printf("|V'_%d_%d| = %d |E_%d| = %lu\n", partition_id, s_i, local_partition_offset[s_i+1] - local_partition_offset[s_i], partition_id, sub_part_out_edges);
        #endif
      }
    }

    VertexId * filtered_out_degree = alloc_vertex_array<VertexId>();
    for (VertexId v_i=partition_overlap_offset[beg_id];v_i<partition_overlap_offset[end_id];v_i++) {
      filtered_out_degree[v_i] = out_degree[v_i];
    }
    numa_free(out_degree, sizeof(VertexId) * vertices);
    out_degree = filtered_out_degree;
    in_degree = out_degree;

    int * buffered_edges = new int [partitions];
    std::vector<char> * send_buffer = new std::vector<char> [partitions];
    for (int i=0;i<partitions;i++) {
      send_buffer[i].resize(edge_unit_size * CHUNKSIZE);
    }
    EdgeUnit<EdgeData> * recv_buffer = new EdgeUnit<EdgeData> [CHUNKSIZE];

    // constructing symmetric edges
    EdgeId recv_outgoing_edges = 0;
    outgoing_edges = new EdgeId [sockets];
    outgoing_adj_index = new EdgeId* [sockets];
    outgoing_adj_list = new AdjUnit<EdgeData>* [sockets];
    outgoing_adj_bitmap = new Bitmap * [sockets];
    for (int s_i=0;s_i<sockets;s_i++) {
      outgoing_adj_bitmap[s_i] = new Bitmap (vertices);
      outgoing_adj_bitmap[s_i]->clear();
      outgoing_adj_index[s_i] = (EdgeId*)numa_alloc_onnode(sizeof(EdgeId) * (vertices+1), s_i);
    }
    {
      std::thread recv_thread_dst([&](){
        int finished_count = 0;
        MPI_Status recv_status;
        while (finished_count < partitions) {
          MPI_Probe(MPI_ANY_SOURCE, ShuffleGraph, MPI_COMM_WORLD, &recv_status);
          int i = recv_status.MPI_SOURCE;
          assert(recv_status.MPI_TAG == ShuffleGraph && i >=0 && i < partitions);
          int recv_bytes;
          MPI_Get_count(&recv_status, MPI_CHAR, &recv_bytes);
          if (recv_bytes==1) {
            finished_count += 1;
            char c;
            MPI_Recv(&c, 1, MPI_CHAR, i, ShuffleGraph, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            continue;
          }
          assert(recv_bytes % edge_unit_size == 0);
          int recv_edges = recv_bytes / edge_unit_size;
          MPI_Recv(recv_buffer, edge_unit_size * recv_edges, MPI_CHAR, i, ShuffleGraph, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
          // #pragma omp parallel for
          for (EdgeId e_i=0;e_i<recv_edges;e_i++) {
            VertexId src = recv_buffer[e_i].src;
            VertexId dst = recv_buffer[e_i].dst;
            int dst_part = get_local_partition_id(dst);
            if (!outgoing_adj_bitmap[dst_part]->get_bit(src)) {
              outgoing_adj_bitmap[dst_part]->set_bit(src);
              outgoing_adj_index[dst_part][src] = 0;
            }
            __sync_fetch_and_add(&outgoing_adj_index[dst_part][src], 1);
          }
          recv_outgoing_edges += recv_edges;
        }
      });
      for (int i=0;i<partitions;i++) {
        buffered_edges[i] = 0;
      }
      assert(lseek(fin, read_offset, SEEK_SET)==read_offset);
      read_bytes = 0;
      while (read_bytes < bytes_to_read) {
        long curr_read_bytes;
        if (bytes_to_read - read_bytes > edge_unit_size * CHUNKSIZE) {
          curr_read_bytes = read(fin, read_edge_buffer, edge_unit_size * CHUNKSIZE);
        } else {
          curr_read_bytes = read(fin, read_edge_buffer, bytes_to_read - read_bytes);
        }
        assert(curr_read_bytes>=0);
        read_bytes += curr_read_bytes;
        EdgeId curr_read_edges = curr_read_bytes / edge_unit_size;
        for (EdgeId e_i=0;e_i<curr_read_edges;e_i++) {
          VertexId dst = read_edge_buffer[e_i].dst;
          int overlap_id = get_overlap_id(dst);
          int i = (overlap_id-1)/2;
          if( overlap_id==0) i=partitions-1;
          do {
            memcpy(send_buffer[i].data() + edge_unit_size * buffered_edges[i], &read_edge_buffer[e_i], edge_unit_size);
            buffered_edges[i] += 1;
            if (buffered_edges[i] == CHUNKSIZE) {
              MPI_Send(send_buffer[i].data(), edge_unit_size * buffered_edges[i], MPI_CHAR, i, ShuffleGraph, MPI_COMM_WORLD);
              buffered_edges[i] = 0;
            }
            if(overlap_id==0 && i!=0) i=0;
            else i++;
          } while(i <= overlap_id/2);
        }
        for (EdgeId e_i=0;e_i<curr_read_edges;e_i++) {
          // std::swap(read_edge_buffer[e_i].src, read_edge_buffer[e_i].dst);
          VertexId tmp = read_edge_buffer[e_i].src;
          read_edge_buffer[e_i].src = read_edge_buffer[e_i].dst;
          read_edge_buffer[e_i].dst = tmp;
        }
        for (EdgeId e_i=0;e_i<curr_read_edges;e_i++) {
          VertexId dst = read_edge_buffer[e_i].dst;
          int overlap_id = get_overlap_id(dst);
          int i = (overlap_id-1)/2;
          if( overlap_id==0) i=partitions-1;
          do {
            memcpy(send_buffer[i].data() + edge_unit_size * buffered_edges[i], &read_edge_buffer[e_i], edge_unit_size);
            buffered_edges[i] += 1;
            if (buffered_edges[i] == CHUNKSIZE) {
              MPI_Send(send_buffer[i].data(), edge_unit_size * buffered_edges[i], MPI_CHAR, i, ShuffleGraph, MPI_COMM_WORLD);
              buffered_edges[i] = 0;
            }
            if(overlap_id==0 && i!=0) i=0;
            else i++;
          } while(i <= overlap_id/2);
        }
      }
      for (int i=0;i<partitions;i++) {
        if (buffered_edges[i]==0) continue;
        MPI_Send(send_buffer[i].data(), edge_unit_size * buffered_edges[i], MPI_CHAR, i, ShuffleGraph, MPI_COMM_WORLD);
        buffered_edges[i] = 0;
      }
      for (int i=0;i<partitions;i++) {
        char c = 0;
        MPI_Send(&c, 1, MPI_CHAR, i, ShuffleGraph, MPI_COMM_WORLD);
      }
      recv_thread_dst.join();
      #ifdef PRINT_DEBUG_MESSAGES
      printf("machine(%d) got %lu symmetric edges\n", partition_id, recv_outgoing_edges);
      #endif
    }
    compressed_outgoing_adj_vertices = new VertexId [sockets];
    compressed_outgoing_adj_index = new CompressedAdjIndexUnit * [sockets];
    for (int s_i=0;s_i<sockets;s_i++) {
      outgoing_edges[s_i] = 0;
      compressed_outgoing_adj_vertices[s_i] = 0;
      for (VertexId v_i=0;v_i<vertices;v_i++) {
        if (outgoing_adj_bitmap[s_i]->get_bit(v_i)) {
          outgoing_edges[s_i] += outgoing_adj_index[s_i][v_i];
          compressed_outgoing_adj_vertices[s_i] += 1;
        }
      }
      compressed_outgoing_adj_index[s_i] = (CompressedAdjIndexUnit*)numa_alloc_onnode( sizeof(CompressedAdjIndexUnit) * (compressed_outgoing_adj_vertices[s_i] + 1) , s_i );
      compressed_outgoing_adj_index[s_i][0].index = 0;
      EdgeId last_e_i = 0;
      compressed_outgoing_adj_vertices[s_i] = 0;
      for (VertexId v_i=0;v_i<vertices;v_i++) {
        if (outgoing_adj_bitmap[s_i]->get_bit(v_i)) {
          outgoing_adj_index[s_i][v_i] = last_e_i + outgoing_adj_index[s_i][v_i];
          last_e_i = outgoing_adj_index[s_i][v_i];
          compressed_outgoing_adj_index[s_i][compressed_outgoing_adj_vertices[s_i]].vertex = v_i;
          compressed_outgoing_adj_vertices[s_i] += 1;
          compressed_outgoing_adj_index[s_i][compressed_outgoing_adj_vertices[s_i]].index = last_e_i;
        }
      }
      for (VertexId p_v_i=0;p_v_i<compressed_outgoing_adj_vertices[s_i];p_v_i++) {
        VertexId v_i = compressed_outgoing_adj_index[s_i][p_v_i].vertex;
        outgoing_adj_index[s_i][v_i] = compressed_outgoing_adj_index[s_i][p_v_i].index;
        outgoing_adj_index[s_i][v_i+1] = compressed_outgoing_adj_index[s_i][p_v_i+1].index;
      }
      #ifdef PRINT_DEBUG_MESSAGES
      printf("part(%d) E_%d has %lu symmetric edges\n", partition_id, s_i, outgoing_edges[s_i]);
      #endif
      outgoing_adj_list[s_i] = (AdjUnit<EdgeData>*)numa_alloc_onnode(unit_size * outgoing_edges[s_i], s_i);
    }
    {
      std::thread recv_thread_dst([&](){
        int finished_count = 0;
        MPI_Status recv_status;
        while (finished_count < partitions) {
          MPI_Probe(MPI_ANY_SOURCE, ShuffleGraph, MPI_COMM_WORLD, &recv_status);
          int i = recv_status.MPI_SOURCE;
          assert(recv_status.MPI_TAG == ShuffleGraph && i >=0 && i < partitions);
          int recv_bytes;
          MPI_Get_count(&recv_status, MPI_CHAR, &recv_bytes);
          if (recv_bytes==1) {
            finished_count += 1;
            char c;
            MPI_Recv(&c, 1, MPI_CHAR, i, ShuffleGraph, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            continue;
          }
          assert(recv_bytes % edge_unit_size == 0);
          int recv_edges = recv_bytes / edge_unit_size;
          MPI_Recv(recv_buffer, edge_unit_size * recv_edges, MPI_CHAR, i, ShuffleGraph, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
          #pragma omp parallel for
          for (EdgeId e_i=0;e_i<recv_edges;e_i++) {
            VertexId src = recv_buffer[e_i].src;
            VertexId dst = recv_buffer[e_i].dst;
            int dst_part = get_local_partition_id(dst);
            EdgeId pos = __sync_fetch_and_add(&outgoing_adj_index[dst_part][src], 1);
            outgoing_adj_list[dst_part][pos].neighbour = dst;
            if (!std::is_same<EdgeData, Empty>::value) {
              outgoing_adj_list[dst_part][pos].edge_data = recv_buffer[e_i].edge_data;
            }
          }
        }
      });
      for (int i=0;i<partitions;i++) {
        buffered_edges[i] = 0;
      }
      assert(lseek(fin, read_offset, SEEK_SET)==read_offset);
      read_bytes = 0;
      while (read_bytes < bytes_to_read) {
        long curr_read_bytes;
        if (bytes_to_read - read_bytes > edge_unit_size * CHUNKSIZE) {
          curr_read_bytes = read(fin, read_edge_buffer, edge_unit_size * CHUNKSIZE);
        } else {
          curr_read_bytes = read(fin, read_edge_buffer, bytes_to_read - read_bytes);
        }
        assert(curr_read_bytes>=0);
        read_bytes += curr_read_bytes;
        EdgeId curr_read_edges = curr_read_bytes / edge_unit_size;
        for (EdgeId e_i=0;e_i<curr_read_edges;e_i++) {
          VertexId dst = read_edge_buffer[e_i].dst;
          int overlap_id = get_overlap_id(dst);
          int i = (overlap_id-1)/2;
          if( overlap_id==0) i=partitions-1;
          do {
            memcpy(send_buffer[i].data() + edge_unit_size * buffered_edges[i], &read_edge_buffer[e_i], edge_unit_size);
            buffered_edges[i] += 1;
            if (buffered_edges[i] == CHUNKSIZE) {
              MPI_Send(send_buffer[i].data(), edge_unit_size * buffered_edges[i], MPI_CHAR, i, ShuffleGraph, MPI_COMM_WORLD);
              buffered_edges[i] = 0;
            }
            if(overlap_id==0 && i!=0) i=0;
            else i++;
          } while(i <= overlap_id/2);
        }
        for (EdgeId e_i=0;e_i<curr_read_edges;e_i++) {
          // std::swap(read_edge_buffer[e_i].src, read_edge_buffer[e_i].dst);
          VertexId tmp = read_edge_buffer[e_i].src;
          read_edge_buffer[e_i].src = read_edge_buffer[e_i].dst;
          read_edge_buffer[e_i].dst = tmp;
        }
        for (EdgeId e_i=0;e_i<curr_read_edges;e_i++) {
          VertexId dst = read_edge_buffer[e_i].dst;
          int overlap_id = get_overlap_id(dst);
          int i = (overlap_id-1)/2;
          if( overlap_id==0) i=partitions-1;
          do {
            memcpy(send_buffer[i].data() + edge_unit_size * buffered_edges[i], &read_edge_buffer[e_i], edge_unit_size);
            buffered_edges[i] += 1;
            if (buffered_edges[i] == CHUNKSIZE) {
              MPI_Send(send_buffer[i].data(), edge_unit_size * buffered_edges[i], MPI_CHAR, i, ShuffleGraph, MPI_COMM_WORLD);
              buffered_edges[i] = 0;
            }
            if(overlap_id==0 && i!=0) i=0;
            else i++;
          } while(i <= overlap_id/2);
        }
      }
      for (int i=0;i<partitions;i++) {
        if (buffered_edges[i]==0) continue;
        MPI_Send(send_buffer[i].data(), edge_unit_size * buffered_edges[i], MPI_CHAR, i, ShuffleGraph, MPI_COMM_WORLD);
        buffered_edges[i] = 0;
      }
      for (int i=0;i<partitions;i++) {
        char c = 0;
        MPI_Send(&c, 1, MPI_CHAR, i, ShuffleGraph, MPI_COMM_WORLD);
      }
      recv_thread_dst.join();
    }
    for (int s_i=0;s_i<sockets;s_i++) {
      for (VertexId p_v_i=0;p_v_i<compressed_outgoing_adj_vertices[s_i];p_v_i++) {
        VertexId v_i = compressed_outgoing_adj_index[s_i][p_v_i].vertex;
        outgoing_adj_index[s_i][v_i] = compressed_outgoing_adj_index[s_i][p_v_i].index;
        outgoing_adj_index[s_i][v_i+1] = compressed_outgoing_adj_index[s_i][p_v_i+1].index;
      }
    }
    MPI_Barrier(MPI_COMM_WORLD);

    incoming_edges = outgoing_edges;
    incoming_adj_index = outgoing_adj_index;
    incoming_adj_list = outgoing_adj_list;
    incoming_adj_bitmap = outgoing_adj_bitmap;
    compressed_incoming_adj_vertices = compressed_outgoing_adj_vertices;
    compressed_incoming_adj_index = compressed_outgoing_adj_index;
    MPI_Barrier(MPI_COMM_WORLD);

    delete [] buffered_edges;
    delete [] send_buffer;
    delete [] read_edge_buffer;
    delete [] recv_buffer;
    close(fin);

    stepsize=new VertexId * [partitions];
    steppointer=new VertexId [partitions];
    for(int i=0;i<partitions;i++){
      steppointer[i]=10;
      stepsize[i] = new VertexId [4];
      stepsize[i][0]=((i==0?vertices:partition_offset[i])-u_partition_overlap_offset[i*2])/10;
      stepsize[i][1]=((i==0?vertices:partition_offset[i])-u_partition_overlap_offset[i*2])-stepsize[i][0]*9;
      stepsize[i][3]=(u_partition_overlap_offset[i*2+1]-partition_offset[i])/10;
      stepsize[i][2]=(u_partition_overlap_offset[i*2+1]-partition_offset[i])-stepsize[i][3]*9;
    }
    
    tune_chunks();
    tuned_chunks_sparse = tuned_chunks_dense;

    prep_time += MPI_Wtime();

    #ifdef PRINT_DEBUG_MESSAGES
    if (partition_id==0) {
      printf("preprocessing cost: %.2lf (s)\n", prep_time);
    }
    #endif
  }

  // transpose the graph
  void transpose() {
    std::swap(out_degree, in_degree);
    std::swap(outgoing_edges, incoming_edges);
    std::swap(outgoing_adj_index, incoming_adj_index);
    std::swap(outgoing_adj_bitmap, incoming_adj_bitmap);
    std::swap(outgoing_adj_list, incoming_adj_list);
    std::swap(tuned_chunks_dense, tuned_chunks_sparse);
    std::swap(compressed_outgoing_adj_vertices, compressed_incoming_adj_vertices);
    std::swap(compressed_outgoing_adj_index, compressed_incoming_adj_index);
  }

  // load a directed graph from path
  // load a directed graph from path

  void load_directed(std::string path, VertexId org_vertices) {
    double prep_time = 0;
    prep_time -= MPI_Wtime();

    symmetric = false;

    MPI_Datatype vid_t = get_mpi_data_type<VertexId>();
    this->org_vertices=org_vertices;
    this->vertices = (org_vertices/PAGESIZE + 1) * PAGESIZE;
    long total_bytes = file_size(path.c_str());
    this->edges = total_bytes / edge_unit_size;
    #ifdef PRINT_DEBUG_MESSAGES
    if (partition_id==0) {
      printf("|V| = %u, |E| = %lu\n", vertices, edges);
    }
    #endif

    EdgeId read_edges = edges / partitions;
    if (partition_id==partitions-1) {
      read_edges += edges % partitions;
    }
    long bytes_to_read = edge_unit_size * read_edges;
    long read_offset = edge_unit_size * (edges / partitions * partition_id);
    long read_bytes;
    int fin = open(path.c_str(), O_RDONLY);
    EdgeUnit<EdgeData> * read_edge_buffer = new EdgeUnit<EdgeData> [CHUNKSIZE];

    out_degree = alloc_interleaved_vertex_array<VertexId>();
    for (VertexId v_i=0;v_i<vertices;v_i++) {
      out_degree[v_i] = 0;
    }
    assert(lseek(fin, read_offset, SEEK_SET)==read_offset);
    read_bytes = 0;
    while (read_bytes < bytes_to_read) {
      long curr_read_bytes;
      if (bytes_to_read - read_bytes > edge_unit_size * CHUNKSIZE) {
        curr_read_bytes = read(fin, read_edge_buffer, edge_unit_size * CHUNKSIZE);
      } else {
        curr_read_bytes = read(fin, read_edge_buffer, bytes_to_read - read_bytes);
      }
      assert(curr_read_bytes>=0);
      read_bytes += curr_read_bytes;
      EdgeId curr_read_edges = curr_read_bytes / edge_unit_size;
      #pragma omp parallel for
      for (EdgeId e_i=0;e_i<curr_read_edges;e_i++) {
        VertexId src = read_edge_buffer[e_i].src;
        VertexId dst = read_edge_buffer[e_i].dst;
        __sync_fetch_and_add(&out_degree[src], 1);
      }
    }
    MPI_Allreduce(MPI_IN_PLACE, out_degree, vertices, vid_t, MPI_SUM, MPI_COMM_WORLD);

    // locality-aware chunking
    partition_offset = new VertexId [partitions + 1];
    new_partition_offset = new VertexId [partitions + 1];
    partition_overlap_offset = new VertexId [partitions * 2];
    u_partition_overlap_offset = new VertexId [partitions * 2];

    partition_offset[0] = 0;
    partition_overlap_offset[0] = 0;
    EdgeId remained_amount = edges + EdgeId(vertices) * alpha;
    for (int i=0;i<partitions;i++) {
      VertexId remained_partitions = partitions - i;
      EdgeId expected_chunk_size = remained_amount / remained_partitions;
      EdgeId expected_overlap_size = expected_chunk_size * over_rate;
      if (remained_partitions==1) {
        partition_offset[i+1] = vertices;
        EdgeId got_edges = 0;
        bool beg_tag=true;
        for (VertexId v_i=partition_offset[i];v_i<vertices;v_i++) {
          got_edges += out_degree[v_i] + alpha;
          if (beg_tag && got_edges > expected_overlap_size){
            partition_overlap_offset[2*i+1] = v_i;
            beg_tag = false;
          } else if (got_edges > expected_chunk_size - expected_overlap_size){
            partition_overlap_offset[0]=v_i;
            break;
          }
        }
        partition_overlap_offset[2*i+1] = (partition_overlap_offset[2*i+1]) / PAGESIZE * PAGESIZE;
        partition_overlap_offset[0] = (vertices - partition_overlap_offset[0]) / PAGESIZE * PAGESIZE;
        partition_overlap_offset[0] = 0 - partition_overlap_offset[0];
        u_partition_overlap_offset[2*i+1] = partition_overlap_offset[2*i+1];
        u_partition_overlap_offset[0] = partition_overlap_offset[0] + vertices;
      } else {
        EdgeId got_edges = 0;
        bool beg_tag=true,mid_tag=true;
        for (VertexId v_i=partition_offset[i];v_i<vertices;v_i++) {
          got_edges += out_degree[v_i] + alpha;
          if (beg_tag && got_edges > expected_overlap_size){
            partition_overlap_offset[2*i+1] = v_i;
            beg_tag = false;
          }
          else if (mid_tag && got_edges > expected_chunk_size - expected_overlap_size){
            partition_overlap_offset[2*i+2] = v_i;
            mid_tag = false;
          }
          else if (got_edges > expected_chunk_size){
            partition_offset[i+1] = v_i;
            break;
          }
        }
        partition_offset[i+1] = (partition_offset[i+1]) / PAGESIZE * PAGESIZE; // aligned with pages
        partition_overlap_offset[2*i+1] = (partition_overlap_offset[2*i+1]) / PAGESIZE * PAGESIZE;
        partition_overlap_offset[2*i+2] = (partition_overlap_offset[2*i+2]) / PAGESIZE * PAGESIZE;
        if(partition_overlap_offset[2*i+2]<partition_overlap_offset[2*i+1]){
          VertexId tmp = partition_overlap_offset[2*i+1];
          partition_overlap_offset[2*i+1] = partition_overlap_offset[2*i+2];
          partition_overlap_offset[2*i+2] = tmp;
        }
        u_partition_overlap_offset[2*i+1] = partition_overlap_offset[2*i+1];
        u_partition_overlap_offset[2*i+2] = partition_overlap_offset[2*i+2];
      }
      for (VertexId v_i=partition_offset[i];v_i<partition_offset[i+1];v_i++) {
        remained_amount -= out_degree[v_i] + alpha;
      }
    }
    partition_overlap_offset[1]=(vertices + partition_overlap_offset[1] - partition_overlap_offset[partitions*2-2]) / PAGESIZE * PAGESIZE +  partition_overlap_offset[partitions*2-2];
    u_partition_overlap_offset[1] = partition_overlap_offset[1] - vertices;
    assert(partition_offset[partitions]==vertices);
    owned_vertices = partition_overlap_offset[end_id] - partition_overlap_offset[beg_id];

    EdgeId p_get_edge = 0;
    for (int i=0;i<partitions;i++){
      p_get_edge = 0;
      for (VertexId v_i=partition_offset[i];v_i<u_partition_overlap_offset[i*2+1];v_i++)p_get_edge += out_degree[v_i];
      if(partition_id==0)printf("P %d-%d(%d) gets edge %lu\n",i,0,u_partition_overlap_offset[i*2+1],p_get_edge);
      p_get_edge = 0;      
      for (VertexId v_i=u_partition_overlap_offset[i*2+1];v_i<u_partition_overlap_offset[(i*2+2)%(partitions*2)];v_i++)p_get_edge += out_degree[v_i];
      if(partition_id==0)printf("P %d-%d(%d) gets edge %lu\n",i,1,u_partition_overlap_offset[(i*2+2)%(partitions*2)],p_get_edge);
      p_get_edge = 0;
      for (VertexId v_i=u_partition_overlap_offset[(i*2+2)%(partitions*2)];v_i<partition_offset[i+1];v_i++)p_get_edge += out_degree[v_i];

    }

    // check consistency of partition boundaries
    VertexId * global_partition_offset = new VertexId [partitions + 1];
    MPI_Allreduce(partition_offset, global_partition_offset, partitions + 1, vid_t, MPI_MAX, MPI_COMM_WORLD);
    for (int i=0;i<=partitions;i++) {
      assert(partition_offset[i] == global_partition_offset[i]);
    }
    MPI_Allreduce(partition_offset, global_partition_offset, partitions + 1, vid_t, MPI_MIN, MPI_COMM_WORLD);
    for (int i=0;i<=partitions;i++) {
      assert(partition_offset[i] == global_partition_offset[i]);
    }
    #ifdef PRINT_DEBUG_MESSAGES
    if (partition_id==0) {
      for (int i=0;i<partitions;i++) {
        EdgeId part_out_edges = 0;
        for (VertexId v_i=partition_offset[i];v_i<partition_offset[i+1];v_i++) {
          part_out_edges += out_degree[v_i];
        }
        printf("|V'_%d| = %d,%d |E_%d| = %lu\n", i, partition_offset[i], partition_offset[i+1], i, part_out_edges);
        printf("overlap[%d] = %d,overlap[%d] = %d\n",2*i,u_partition_overlap_offset[2*i],2*i+1,u_partition_overlap_offset[2*i+1]);
      }
    }
    #endif
    delete [] global_partition_offset;
    {
      // NUMA-aware sub-chunking
      local_partition_offset = new VertexId [sockets + 1];
      EdgeId part_out_edges = 0;
      for (VertexId v_i=partition_overlap_offset[beg_id];v_i<partition_overlap_offset[end_id];v_i++) {
        VertexId v_mod = v_i;
        if(v_i<0) v_mod = v_i + vertices;
        else if(v_i>=vertices) v_mod = v_i - vertices;
        part_out_edges += out_degree[v_mod];
      }
      local_partition_offset[0] = partition_overlap_offset[beg_id];
      EdgeId remained_amount = part_out_edges + EdgeId(owned_vertices) * alpha;
      for (int s_i=0;s_i<sockets;s_i++) {
        VertexId remained_partitions = sockets - s_i;
        EdgeId expected_chunk_size = remained_amount / remained_partitions;
        if (remained_partitions==1) {
          local_partition_offset[s_i+1] = partition_overlap_offset[end_id];
        } else {
          EdgeId got_edges = 0;
          for (VertexId v_i=local_partition_offset[s_i];v_i<partition_overlap_offset[end_id];v_i++) {
            VertexId v_mod = v_i;
            if(v_i<0) v_mod = v_i + vertices;
            else if(v_i>=vertices) v_mod = v_i - vertices;
            got_edges += out_degree[v_mod] + alpha;
            if (got_edges > expected_chunk_size) {
              local_partition_offset[s_i+1] = v_i;
              break;
            }
          }
          local_partition_offset[s_i+1] = (local_partition_offset[s_i+1]-local_partition_offset[s_i]) / PAGESIZE * PAGESIZE + local_partition_offset[s_i]; // aligned with pages
        }
        EdgeId sub_part_out_edges = 0;
        for (VertexId v_i=local_partition_offset[s_i];v_i<local_partition_offset[s_i+1];v_i++) {
          VertexId v_mod = v_i;
          if(v_i<0) v_mod = v_i + vertices;
          else if(v_i>=vertices) v_mod = v_i - vertices;
          remained_amount -= out_degree[v_mod] + alpha;
          sub_part_out_edges += out_degree[v_mod];
        }
        #ifdef PRINT_DEBUG_MESSAGES
        printf("|V'_%d_%d| = %d,%d |E^dense_%d_%d| = %d\n", partition_id, s_i, local_partition_offset[s_i], local_partition_offset[s_i+1], partition_id, s_i, sub_part_out_edges);
        #endif
      }
    }

    VertexId * filtered_out_degree = alloc_vertex_array<VertexId>();
    for (VertexId v_i=partition_overlap_offset[beg_id];v_i<partition_overlap_offset[end_id];v_i++) {
      VertexId v_mod = v_i;
      if(v_i<0) v_mod = v_i + vertices;
      else if(v_i>=vertices) v_mod = v_i - vertices;
      filtered_out_degree[v_mod] = out_degree[v_mod];
    }
    numa_free(out_degree, sizeof(VertexId) * vertices);
    out_degree = filtered_out_degree;
    in_degree = alloc_vertex_array<VertexId>();
    //edit @ 09114 by wyd
    for (VertexId v_i=partition_overlap_offset[beg_id];v_i<partition_overlap_offset[end_id];v_i++) {
      VertexId v_mod = v_i;
      if(v_i<0) v_mod = v_i + vertices;
      else if(v_i>=vertices) v_mod = v_i - vertices;
      in_degree[v_mod] = 0;
    }
    //edit @ 0914

    int * buffered_edges = new int [partitions];
    std::vector<char> * send_buffer = new std::vector<char> [partitions];
    for (int i=0;i<partitions;i++) {
      send_buffer[i].resize(edge_unit_size * CHUNKSIZE);
    }
    EdgeUnit<EdgeData> * recv_buffer = new EdgeUnit<EdgeData> [CHUNKSIZE];

    EdgeId recv_outgoing_edges = 0;
    outgoing_edges = new EdgeId [sockets];
    outgoing_adj_index = new EdgeId* [sockets];
    outgoing_adj_list = new AdjUnit<EdgeData>* [sockets];
    outgoing_adj_bitmap = new Bitmap * [sockets];
    for (int s_i=0;s_i<sockets;s_i++) {
      outgoing_adj_bitmap[s_i] = new Bitmap (vertices);
      outgoing_adj_bitmap[s_i]->clear();
      outgoing_adj_index[s_i] = (EdgeId*)numa_alloc_onnode(sizeof(EdgeId) * (vertices+1), s_i);
    }
    {
      std::thread recv_thread_dst([&](){
        int finished_count = 0;
        MPI_Status recv_status;
        while (finished_count < partitions) {
          MPI_Probe(MPI_ANY_SOURCE, ShuffleGraph, MPI_COMM_WORLD, &recv_status);
          int i = recv_status.MPI_SOURCE;
          assert(recv_status.MPI_TAG == ShuffleGraph && i >=0 && i < partitions);
          int recv_bytes;
          MPI_Get_count(&recv_status, MPI_CHAR, &recv_bytes);
          if (recv_bytes==1) {
            finished_count += 1;
            char c;
            MPI_Recv(&c, 1, MPI_CHAR, i, ShuffleGraph, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            continue;
          }
          assert(recv_bytes % edge_unit_size == 0);
          int recv_edges = recv_bytes / edge_unit_size;
          MPI_Recv(recv_buffer, edge_unit_size * recv_edges, MPI_CHAR, i, ShuffleGraph, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
          // #pragma omp parallel for
          for (EdgeId e_i=0;e_i<recv_edges;e_i++) {
            VertexId src = recv_buffer[e_i].src;
            VertexId dst = recv_buffer[e_i].dst;
            //assert(dst >= partition_overlap_offset[beg_id] && dst < partition_overlap_offset[end_id]);
            int dst_part = get_local_partition_id(dst);
            if (!outgoing_adj_bitmap[dst_part]->get_bit(src)) {
              outgoing_adj_bitmap[dst_part]->set_bit(src);
              outgoing_adj_index[dst_part][src] = 0;
            }
            __sync_fetch_and_add(&outgoing_adj_index[dst_part][src], 1);
            __sync_fetch_and_add(&in_degree[dst], 1);
          }
          recv_outgoing_edges += recv_edges;
        }
      });
      for (int i=0;i<partitions;i++) {
        buffered_edges[i] = 0;
      }
      assert(lseek(fin, read_offset, SEEK_SET)==read_offset);
      read_bytes = 0;
      while (read_bytes < bytes_to_read) {
        long curr_read_bytes;
        if (bytes_to_read - read_bytes > edge_unit_size * CHUNKSIZE) {
          curr_read_bytes = read(fin, read_edge_buffer, edge_unit_size * CHUNKSIZE);
        } else {
          curr_read_bytes = read(fin, read_edge_buffer, bytes_to_read - read_bytes);
        }
        assert(curr_read_bytes>=0);
        read_bytes += curr_read_bytes;
        EdgeId curr_read_edges = curr_read_bytes / edge_unit_size;
        for (EdgeId e_i=0;e_i<curr_read_edges;e_i++){
          VertexId dst = read_edge_buffer[e_i].dst;
          // wyd@29/7/17 memcpy to another owner
          int overlap_id = get_overlap_id(dst);
          int i = (overlap_id-1)/2;
          if( overlap_id==0) i=partitions-1;
          do {
            memcpy(send_buffer[i].data() + edge_unit_size * buffered_edges[i], &read_edge_buffer[e_i], edge_unit_size);
            buffered_edges[i] += 1;
            if (buffered_edges[i] == CHUNKSIZE) {
              MPI_Send(send_buffer[i].data(), edge_unit_size * buffered_edges[i], MPI_CHAR, i, ShuffleGraph, MPI_COMM_WORLD);
              buffered_edges[i] = 0;
            }
            if(overlap_id==0 && i!=0) i=0;
            else i++;
          }while(i <= overlap_id/2);
          // wyd@29/7/17 memcpy to another owner
        }
      }
      for (int i=0;i<partitions;i++) {
        if (buffered_edges[i]==0) continue;
        MPI_Send(send_buffer[i].data(), edge_unit_size * buffered_edges[i], MPI_CHAR, i, ShuffleGraph, MPI_COMM_WORLD);
        buffered_edges[i] = 0;
      }
      for (int i=0;i<partitions;i++) {
        char c = 0;
        MPI_Send(&c, 1, MPI_CHAR, i, ShuffleGraph, MPI_COMM_WORLD);
      }
      recv_thread_dst.join();
      #ifdef PRINT_DEBUG_MESSAGES
      printf("machine(%d) got %lu sparse mode edges\n", partition_id, recv_outgoing_edges);
      #endif
    }
    compressed_outgoing_adj_vertices = new VertexId [sockets];
    compressed_outgoing_adj_index = new CompressedAdjIndexUnit * [sockets];
    for (int s_i=0;s_i<sockets;s_i++) {
      outgoing_edges[s_i] = 0;
      compressed_outgoing_adj_vertices[s_i] = 0;
      for (VertexId v_i=0;v_i<vertices;v_i++) {
        if (outgoing_adj_bitmap[s_i]->get_bit(v_i)) {
          outgoing_edges[s_i] += outgoing_adj_index[s_i][v_i];
          compressed_outgoing_adj_vertices[s_i] += 1;
        }
      }
      compressed_outgoing_adj_index[s_i] = (CompressedAdjIndexUnit*)numa_alloc_onnode( sizeof(CompressedAdjIndexUnit) * (compressed_outgoing_adj_vertices[s_i] + 1) , s_i );
      compressed_outgoing_adj_index[s_i][0].index = 0;
      EdgeId last_e_i = 0;
      compressed_outgoing_adj_vertices[s_i] = 0;
      for (VertexId v_i=0;v_i<vertices;v_i++) {
        if (outgoing_adj_bitmap[s_i]->get_bit(v_i)) {
          outgoing_adj_index[s_i][v_i] = last_e_i + outgoing_adj_index[s_i][v_i];
          last_e_i = outgoing_adj_index[s_i][v_i];
          compressed_outgoing_adj_index[s_i][compressed_outgoing_adj_vertices[s_i]].vertex = v_i;
          compressed_outgoing_adj_vertices[s_i] += 1;
          compressed_outgoing_adj_index[s_i][compressed_outgoing_adj_vertices[s_i]].index = last_e_i;
        }
      }
      for (VertexId p_v_i=0;p_v_i<compressed_outgoing_adj_vertices[s_i];p_v_i++) {
        VertexId v_i = compressed_outgoing_adj_index[s_i][p_v_i].vertex;
        outgoing_adj_index[s_i][v_i] = compressed_outgoing_adj_index[s_i][p_v_i].index;
        outgoing_adj_index[s_i][v_i+1] = compressed_outgoing_adj_index[s_i][p_v_i+1].index;
      }
      #ifdef PRINT_DEBUG_MESSAGES
      printf("part(%d) E_%d has %lu sparse mode edges\n", partition_id, s_i, outgoing_edges[s_i]);
      #endif
      outgoing_adj_list[s_i] = (AdjUnit<EdgeData>*)numa_alloc_onnode(unit_size * outgoing_edges[s_i], s_i);
    }
    {
      std::thread recv_thread_dst([&](){
        int finished_count = 0;
        MPI_Status recv_status;
        while (finished_count < partitions) {
          MPI_Probe(MPI_ANY_SOURCE, ShuffleGraph, MPI_COMM_WORLD, &recv_status);
          int i = recv_status.MPI_SOURCE;
          assert(recv_status.MPI_TAG == ShuffleGraph && i >=0 && i < partitions);
          int recv_bytes;
          MPI_Get_count(&recv_status, MPI_CHAR, &recv_bytes);
          if (recv_bytes==1) {
            finished_count += 1;
            char c;
            MPI_Recv(&c, 1, MPI_CHAR, i, ShuffleGraph, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            continue;
          }
          assert(recv_bytes % edge_unit_size == 0);
          int recv_edges = recv_bytes / edge_unit_size;
          MPI_Recv(recv_buffer, edge_unit_size * recv_edges, MPI_CHAR, i, ShuffleGraph, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
          #pragma omp parallel for
          for (EdgeId e_i=0;e_i<recv_edges;e_i++) {
            VertexId src = recv_buffer[e_i].src;
            VertexId dst = recv_buffer[e_i].dst;
            //assert(dst >= partition_overlap_offset[beg_id] && dst < partition_overlap_offset[end_id]);
            int dst_part = get_local_partition_id(dst);
            EdgeId pos = __sync_fetch_and_add(&outgoing_adj_index[dst_part][src], 1);
            // wyd add if 0302
            outgoing_adj_list[dst_part][pos].neighbour = dst;
            if (!std::is_same<EdgeData, Empty>::value) {
              outgoing_adj_list[dst_part][pos].edge_data = recv_buffer[e_i].edge_data;
            }
          }
        }
      });
      for (int i=0;i<partitions;i++) {
        buffered_edges[i] = 0;
      }
      assert(lseek(fin, read_offset, SEEK_SET)==read_offset);
      read_bytes = 0;
      while (read_bytes < bytes_to_read) {
        long curr_read_bytes;
        if (bytes_to_read - read_bytes > edge_unit_size * CHUNKSIZE) {
          curr_read_bytes = read(fin, read_edge_buffer, edge_unit_size * CHUNKSIZE);
        } else {
          curr_read_bytes = read(fin, read_edge_buffer, bytes_to_read - read_bytes);
        }
        assert(curr_read_bytes>=0);
        read_bytes += curr_read_bytes;
        EdgeId curr_read_edges = curr_read_bytes / edge_unit_size;
        for (EdgeId e_i=0;e_i<curr_read_edges;e_i++) {
          VertexId dst = read_edge_buffer[e_i].dst;
          //wyd@28/7/17
          int overlap_id = get_overlap_id(dst);
          int i = (overlap_id-1)/2;
          if( overlap_id==0) i=partitions-1;
          do {
            memcpy(send_buffer[i].data() + edge_unit_size * buffered_edges[i], &read_edge_buffer[e_i], edge_unit_size);
            buffered_edges[i] += 1;
            if (buffered_edges[i] == CHUNKSIZE) {
              MPI_Send(send_buffer[i].data(), edge_unit_size * buffered_edges[i], MPI_CHAR, i, ShuffleGraph, MPI_COMM_WORLD);
              buffered_edges[i] = 0;
            }
            if(overlap_id==0 && i!=0) i=0;
            else i++;
          }while(i <= overlap_id/2);
          //wyd@28/7/17
        }
      }
      for (int i=0;i<partitions;i++) {
        if (buffered_edges[i]==0) continue;
        MPI_Send(send_buffer[i].data(), edge_unit_size * buffered_edges[i], MPI_CHAR, i, ShuffleGraph, MPI_COMM_WORLD);
        buffered_edges[i] = 0;
      }
      for (int i=0;i<partitions;i++) {
        char c = 0;
        MPI_Send(&c, 1, MPI_CHAR, i, ShuffleGraph, MPI_COMM_WORLD);
      }
      recv_thread_dst.join();
    }
    for (int s_i=0;s_i<sockets;s_i++) {
      for (VertexId p_v_i=0;p_v_i<compressed_outgoing_adj_vertices[s_i];p_v_i++) {
        VertexId v_i = compressed_outgoing_adj_index[s_i][p_v_i].vertex;
        outgoing_adj_index[s_i][v_i] = compressed_outgoing_adj_index[s_i][p_v_i].index;
        outgoing_adj_index[s_i][v_i+1] = compressed_outgoing_adj_index[s_i][p_v_i+1].index;
      }
    }
    MPI_Barrier(MPI_COMM_WORLD);

    EdgeId recv_incoming_edges = 0;
    incoming_edges = new EdgeId [sockets];
    incoming_adj_index = new EdgeId* [sockets];
    incoming_adj_list = new AdjUnit<EdgeData>* [sockets];
    incoming_adj_bitmap = new Bitmap * [sockets];
    for (int s_i=0;s_i<sockets;s_i++) {
      incoming_adj_bitmap[s_i] = new Bitmap (vertices);
      incoming_adj_bitmap[s_i]->clear();
      incoming_adj_index[s_i] = (EdgeId*)numa_alloc_onnode(sizeof(EdgeId) * (vertices+1), s_i);
    }
    {
      std::thread recv_thread_src([&](){
        int finished_count = 0;
        MPI_Status recv_status;
        while (finished_count < partitions) {
          MPI_Probe(MPI_ANY_SOURCE, ShuffleGraph, MPI_COMM_WORLD, &recv_status);
          int i = recv_status.MPI_SOURCE;
          assert(recv_status.MPI_TAG == ShuffleGraph && i >=0 && i < partitions);
          int recv_bytes;
          MPI_Get_count(&recv_status, MPI_CHAR, &recv_bytes);
          if (recv_bytes==1) {
            finished_count += 1;
            char c;
            MPI_Recv(&c, 1, MPI_CHAR, i, ShuffleGraph, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            continue;
          }
          assert(recv_bytes % edge_unit_size == 0);
          int recv_edges = recv_bytes / edge_unit_size;
          MPI_Recv(recv_buffer, edge_unit_size * recv_edges, MPI_CHAR, i, ShuffleGraph, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
          // #pragma omp parallel for
          for (EdgeId e_i=0;e_i<recv_edges;e_i++) {
            VertexId src = recv_buffer[e_i].src;
            VertexId dst = recv_buffer[e_i].dst;
            //assert(src >= partition_overlap_offset[beg_id] && src < partition_overlap_offset[end_id]);
            int src_part = get_local_partition_id(src);
            if (!incoming_adj_bitmap[src_part]->get_bit(dst)) {
              incoming_adj_bitmap[src_part]->set_bit(dst);
              incoming_adj_index[src_part][dst] = 0;
            }
            __sync_fetch_and_add(&incoming_adj_index[src_part][dst], 1);
          }
          recv_incoming_edges += recv_edges;
        }
      });
      for (int i=0;i<partitions;i++) {
        buffered_edges[i] = 0;
      }
      assert(lseek(fin, read_offset, SEEK_SET)==read_offset);
      read_bytes = 0;
      while (read_bytes < bytes_to_read) {
        long curr_read_bytes;
        if (bytes_to_read - read_bytes > edge_unit_size * CHUNKSIZE) {
          curr_read_bytes = read(fin, read_edge_buffer, edge_unit_size * CHUNKSIZE);
        } else {
          curr_read_bytes = read(fin, read_edge_buffer, bytes_to_read - read_bytes);
        }
        assert(curr_read_bytes>=0);
        read_bytes += curr_read_bytes;
        EdgeId curr_read_edges = curr_read_bytes / edge_unit_size;
        for (EdgeId e_i=0;e_i<curr_read_edges;e_i++) {
          VertexId src = read_edge_buffer[e_i].src;
          int overlap_id = get_overlap_id(src);
          int i = (overlap_id-1)/2;
          if( overlap_id==0) i=partitions-1;
          do {
            memcpy(send_buffer[i].data() + edge_unit_size * buffered_edges[i], &read_edge_buffer[e_i], edge_unit_size);
            buffered_edges[i] += 1;
            if (buffered_edges[i] == CHUNKSIZE) {
              MPI_Send(send_buffer[i].data(), edge_unit_size * buffered_edges[i], MPI_CHAR, i, ShuffleGraph, MPI_COMM_WORLD);
              buffered_edges[i] = 0;
            }
            if(overlap_id==0 && i!=0) i=0;
            else i++;
          } while(i <= overlap_id/2);
        }
      }
      for (int i=0;i<partitions;i++) {
        if (buffered_edges[i]==0) continue;
        MPI_Send(send_buffer[i].data(), edge_unit_size * buffered_edges[i], MPI_CHAR, i, ShuffleGraph, MPI_COMM_WORLD);
        buffered_edges[i] = 0;
      }
      for (int i=0;i<partitions;i++) {
        char c = 0;
        MPI_Send(&c, 1, MPI_CHAR, i, ShuffleGraph, MPI_COMM_WORLD);
      }
      recv_thread_src.join();
      #ifdef PRINT_DEBUG_MESSAGES
      printf("machine(%d) got %lu dense mode edges\n", partition_id, recv_incoming_edges);
      #endif
    }
    compressed_incoming_adj_vertices = new VertexId [sockets];
    compressed_incoming_adj_index = new CompressedAdjIndexUnit * [sockets];
    for (int s_i=0;s_i<sockets;s_i++) {
      incoming_edges[s_i] = 0;
      compressed_incoming_adj_vertices[s_i] = 0;
      for (VertexId v_i=0;v_i<vertices;v_i++) {
        if (incoming_adj_bitmap[s_i]->get_bit(v_i)) {
          incoming_edges[s_i] += incoming_adj_index[s_i][v_i];
          compressed_incoming_adj_vertices[s_i] += 1;
        }
      }

      compressed_incoming_adj_index[s_i] = (CompressedAdjIndexUnit*)numa_alloc_onnode( sizeof(CompressedAdjIndexUnit) * (compressed_incoming_adj_vertices[s_i] + 1) , s_i );
      compressed_incoming_adj_index[s_i][0].index = 0;
      EdgeId last_e_i = 0;
      compressed_incoming_adj_vertices[s_i] = 0;
      for (VertexId v_i=0;v_i<vertices;v_i++) {
        if (incoming_adj_bitmap[s_i]->get_bit(v_i)) {
          incoming_adj_index[s_i][v_i] = last_e_i + incoming_adj_index[s_i][v_i];
          last_e_i = incoming_adj_index[s_i][v_i];
          compressed_incoming_adj_index[s_i][compressed_incoming_adj_vertices[s_i]].vertex = v_i;
          compressed_incoming_adj_vertices[s_i] += 1;
          compressed_incoming_adj_index[s_i][compressed_incoming_adj_vertices[s_i]].index = last_e_i;
        }
      }
      for (VertexId p_v_i=0;p_v_i<compressed_incoming_adj_vertices[s_i];p_v_i++) {
        VertexId v_i = compressed_incoming_adj_index[s_i][p_v_i].vertex;
        incoming_adj_index[s_i][v_i] = compressed_incoming_adj_index[s_i][p_v_i].index;
        incoming_adj_index[s_i][v_i+1] = compressed_incoming_adj_index[s_i][p_v_i+1].index;
      }
      #ifdef PRINT_DEBUG_MESSAGES
      printf("part(%d) E_%d has %lu dense mode edges\n", partition_id, s_i, incoming_edges[s_i]);
      #endif
      incoming_adj_list[s_i] = (AdjUnit<EdgeData>*)numa_alloc_onnode(unit_size * incoming_edges[s_i], s_i);
    }
    {
      std::thread recv_thread_src([&](){
        int finished_count = 0;
        MPI_Status recv_status;
        //while (finished_count < partitions) {
        for(finished_count=0;finished_count<partitions;){
          //MPI_Probe(MPI_ANY_SOURCE, ShuffleGraph, MPI_COMM_WORLD, &recv_status);
          MPI_Probe(finished_count, ShuffleGraph, MPI_COMM_WORLD, &recv_status);
          int i = recv_status.MPI_SOURCE;
          assert(recv_status.MPI_TAG == ShuffleGraph && i == finished_count);
          int recv_bytes;
          MPI_Get_count(&recv_status, MPI_CHAR, &recv_bytes);
          if (recv_bytes==1) {
            finished_count += 1;
            char c;
            MPI_Recv(&c, 1, MPI_CHAR, i, ShuffleGraph, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            continue;
          }
          assert(recv_bytes % edge_unit_size == 0);
          int recv_edges = recv_bytes / edge_unit_size;
          MPI_Recv(recv_buffer, edge_unit_size * recv_edges, MPI_CHAR, i, ShuffleGraph, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
          //#pragma omp parallel for
          for (EdgeId e_i=0;e_i<recv_edges;e_i++) {
            VertexId src = recv_buffer[e_i].src;
            VertexId dst = recv_buffer[e_i].dst;
            //assert(src >= partition_overlap_offset[beg_id] && src < partition_overlap_offset[end_id]);
            int src_part = get_local_partition_id(src);
            EdgeId pos = __sync_fetch_and_add(&incoming_adj_index[src_part][dst], 1);
            incoming_adj_list[src_part][pos].neighbour = src;
            if (!std::is_same<EdgeData, Empty>::value) {
              incoming_adj_list[src_part][pos].edge_data = recv_buffer[e_i].edge_data;
            }
          }
        }
      });
      for (int i=0;i<partitions;i++) {
        buffered_edges[i] = 0;
      }
      assert(lseek(fin, read_offset, SEEK_SET)==read_offset);
      read_bytes = 0;
      while (read_bytes < bytes_to_read) {
        long curr_read_bytes;
        if (bytes_to_read - read_bytes > edge_unit_size * CHUNKSIZE) {
          curr_read_bytes = read(fin, read_edge_buffer, edge_unit_size * CHUNKSIZE);
        } else {
          curr_read_bytes = read(fin, read_edge_buffer, bytes_to_read - read_bytes);
        }
        assert(curr_read_bytes>=0);
        read_bytes += curr_read_bytes;
        EdgeId curr_read_edges = curr_read_bytes / edge_unit_size;
        for (EdgeId e_i=0;e_i<curr_read_edges;e_i++) {
          VertexId src = read_edge_buffer[e_i].src;
          int overlap_id = get_overlap_id(src);
          int i = (overlap_id-1)/2;
          if( overlap_id==0) i=partitions-1;
          do {
            memcpy(send_buffer[i].data() + edge_unit_size * buffered_edges[i], &read_edge_buffer[e_i], edge_unit_size);
            buffered_edges[i] += 1;
            if (buffered_edges[i] == CHUNKSIZE) {
              MPI_Send(send_buffer[i].data(), edge_unit_size * buffered_edges[i], MPI_CHAR, i, ShuffleGraph, MPI_COMM_WORLD);
              buffered_edges[i] = 0;
            }
            if(overlap_id==0 && i!=0) i=0;
            else i++;
          }while(i <= overlap_id/2);
        }
      }
      for (int i=0;i<partitions;i++) {
        if (buffered_edges[i]==0) continue;
        MPI_Send(send_buffer[i].data(), edge_unit_size * buffered_edges[i], MPI_CHAR, i, ShuffleGraph, MPI_COMM_WORLD);
        buffered_edges[i] = 0;
      }
      for (int i=0;i<partitions;i++) {
        char c = 0;
        MPI_Send(&c, 1, MPI_CHAR, i, ShuffleGraph, MPI_COMM_WORLD);
      }
      recv_thread_src.join();
    }
    for (int s_i=0;s_i<sockets;s_i++) {
      for (VertexId p_v_i=0;p_v_i<compressed_incoming_adj_vertices[s_i];p_v_i++) {
        VertexId v_i = compressed_incoming_adj_index[s_i][p_v_i].vertex;
        incoming_adj_index[s_i][v_i] = compressed_incoming_adj_index[s_i][p_v_i].index;
        incoming_adj_index[s_i][v_i+1] = compressed_incoming_adj_index[s_i][p_v_i+1].index;
          for(int w=incoming_adj_index[s_i][v_i];w<incoming_adj_index[s_i][v_i+1]-1;w++){
            if(incoming_adj_list[s_i][w].neighbour>incoming_adj_list[s_i][w+1].neighbour)
              printf("%d %d %d unsorted %d\n",v_i,incoming_adj_list[s_i][w].neighbour,incoming_adj_list[s_i][w+1].neighbour,partition_id);
          }
      }
    }
    MPI_Barrier(MPI_COMM_WORLD);

    delete [] buffered_edges;
    delete [] send_buffer;
    delete [] read_edge_buffer;
    delete [] recv_buffer;
    close(fin);

    stepsize=new VertexId * [partitions];
    steppointer=new VertexId [partitions];
    for(int i=0;i<partitions;i++){
      steppointer[i]=10;
      stepsize[i] = new VertexId [4]; 
      stepsize[i][0]=((i==0?vertices:partition_offset[i])-u_partition_overlap_offset[i*2])/10/64*64;
      stepsize[i][1]=((i==0?vertices:partition_offset[i])-u_partition_overlap_offset[i*2])-stepsize[i][0]*9;
      stepsize[i][3]=(u_partition_overlap_offset[i*2+1]-partition_offset[i])/10/64*64;
      stepsize[i][2]=(u_partition_overlap_offset[i*2+1]-partition_offset[i])-stepsize[i][3]*9;
    }    
    tune_chunks();

    prep_time += MPI_Wtime();

    #ifdef PRINT_DEBUG_MESSAGES
    if (partition_id==0) {
      printf("preprocessing cost: %.2lf (s)\n", prep_time);
    }
    #endif
  }

  void tune_chunks() {
    tuned_chunks_dense = new ThreadState * [partitions];
    tuned_chunks_currs = new VertexId ** [partitions];
    tuned_chunks_ends = new VertexId ** [partitions];
    int steps=20;

    int current_send_part_id = partition_id;
    for (int step=0;step<partitions;step++) {
      current_send_part_id = (current_send_part_id + 1) % partitions;
      int i = current_send_part_id;
      tuned_chunks_dense[i] = new ThreadState [threads];
      tuned_chunks_currs[i] = new VertexId * [sockets];
      tuned_chunks_ends[i] = new VertexId * [sockets];

      EdgeId remained_edges;
      int remained_partitions;
      VertexId last_p_v_i;
      VertexId end_p_v_i;
      for (int t_i=0;t_i<threads;t_i++) {
        tuned_chunks_dense[i][t_i].status = WORKING;
        int s_i = get_socket_id(t_i);
        int s_j = get_socket_offset(t_i);
        if (s_j==0) {
          tuned_chunks_currs[i][s_i] = new VertexId [steps+1];
          tuned_chunks_ends[i][s_i] = new VertexId [steps+1];
          VertexId p_v_i = 0;
          int pointer=0;
          VertexId partition_beg_index=u_partition_overlap_offset[i*2];
          VertexId partition_mid_index=i==0?vertices:partition_offset[i];
          VertexId partition_end_index=u_partition_overlap_offset[i*2+1];
          VertexId stepsize1=(partition_mid_index-partition_beg_index)/10/64*64;
          VertexId remainsize=0;

          while (p_v_i<compressed_incoming_adj_vertices[s_i]) {
            VertexId v_i = compressed_incoming_adj_index[s_i][p_v_i].vertex;
            if (v_i >= partition_beg_index+remainsize) {
              tuned_chunks_currs[i][s_i][pointer++]=p_v_i;
              remainsize+=stepsize1;
              if(pointer==10)
                break;
            }
            p_v_i++;
          }
          if(i==0)p_v_i=0;
          partition_mid_index=partition_offset[i];
          VertexId stepsize2=(partition_end_index-partition_mid_index)/10/64*64;
          remainsize=0;
          while (p_v_i<compressed_incoming_adj_vertices[s_i]) {
            VertexId v_i = compressed_incoming_adj_index[s_i][p_v_i].vertex;
            if (v_i >= partition_mid_index+remainsize) {
              tuned_chunks_currs[i][s_i][pointer++]=p_v_i;
              if(pointer==11)
                remainsize=partition_end_index-partition_mid_index-stepsize2*9;
              else if(pointer==21)
                break;
              else
                remainsize+=stepsize2;
            }
            p_v_i++;
          }
          last_p_v_i = p_v_i;

          partition_beg_index=u_partition_overlap_offset[(i*2+2)%(partitions*2)];
          partition_mid_index=partition_offset[i+1];
          partition_end_index=u_partition_overlap_offset[(i*2+3)%(partitions*2)];
          stepsize1=(partition_mid_index-partition_beg_index)/10/64*64;
          remainsize=0;
          pointer=0;

          while (p_v_i<compressed_incoming_adj_vertices[s_i]) {
            VertexId v_i = compressed_incoming_adj_index[s_i][p_v_i].vertex;
            if (v_i >= partition_beg_index+remainsize) {
              tuned_chunks_ends[i][s_i][pointer++]=p_v_i;
              if(pointer==10)
                remainsize+=partition_mid_index-partition_beg_index-stepsize1*9;
              else if(pointer==11)
                break;
              else
                remainsize+=stepsize1;
            }
            p_v_i++;
          }
          if(pointer!=11){
            tuned_chunks_ends[i][s_i][pointer++]=p_v_i-1;
            printf("pvi %d %d\n",compressed_incoming_adj_index[s_i][p_v_i].vertex,compressed_incoming_adj_index[s_i][p_v_i-1].vertex);
          }
          if(i==partitions-1)p_v_i=0;
          partition_mid_index=i==partitions-1?0:partition_offset[i+1];
          stepsize2=(partition_end_index-partition_mid_index)/10/64*64;
          remainsize=partition_end_index-partition_mid_index-stepsize2*9;
          while (p_v_i<compressed_incoming_adj_vertices[s_i]) {
            VertexId v_i = compressed_incoming_adj_index[s_i][p_v_i].vertex;
            if (v_i >= partition_mid_index+remainsize) {
              tuned_chunks_ends[i][s_i][pointer++]=p_v_i;
              if(pointer==21)
                break;
              else
                remainsize+=stepsize2;
            }
            p_v_i++;
          }
          end_p_v_i = tuned_chunks_ends[i][s_i][0];
          /*
          if(partition_id==1)
            for(int w_i=0;w_i<21;w_i++)
              printf("node %d s_i=0 curr %d\n",tuned_chunks_currs[i][0][w_i]);
          if(partition_id==1)
            for(int w_i=0;w_i<21;w_i++)
              printf("node %d s_i=0 end %d\n",tuned_chunks_ends[i][0][w_i]);
          */
          remained_edges = 0;
          for (VertexId p_v_i=last_p_v_i;p_v_i<end_p_v_i;p_v_i++) {
            remained_edges += compressed_incoming_adj_index[s_i][p_v_i+1].index - compressed_incoming_adj_index[s_i][p_v_i].index;
            remained_edges += alpha;
          }
        }

        tuned_chunks_dense[i][t_i].curr = last_p_v_i;
        tuned_chunks_dense[i][t_i].end = last_p_v_i;

        remained_partitions = threads_per_socket - s_j;
        EdgeId expected_chunk_size = remained_edges / remained_partitions;
        if (remained_partitions==1) {
          tuned_chunks_dense[i][t_i].end = end_p_v_i;
        } else {
          EdgeId got_edges = 0;
          for (VertexId p_v_i=last_p_v_i;p_v_i<end_p_v_i;p_v_i++) {
            got_edges += compressed_incoming_adj_index[s_i][p_v_i+1].index - compressed_incoming_adj_index[s_i][p_v_i].index + alpha;
            if (got_edges >= expected_chunk_size) {
              tuned_chunks_dense[i][t_i].end = p_v_i;
              last_p_v_i = tuned_chunks_dense[i][t_i].end;
              break;
            }
          }
          got_edges = 0;
          for (VertexId p_v_i=tuned_chunks_dense[i][t_i].curr;p_v_i<tuned_chunks_dense[i][t_i].end;p_v_i++) {
            got_edges += compressed_incoming_adj_index[s_i][p_v_i+1].index - compressed_incoming_adj_index[s_i][p_v_i].index + alpha;
          }
          remained_edges -= got_edges;
        }
        if(s_j==0)
          tuned_chunks_dense[i][t_i].curr = tuned_chunks_currs[i][s_i][10];
        if(s_j==threads_per_socket-1)
          tuned_chunks_dense[i][t_i].end = tuned_chunks_ends[i][s_i][10];
      }
    }
  }

  // process vertices
  template<typename R>
  R process_vertices(std::function<R(VertexId)> process, Bitmap * active) {
    double stream_time = 0;
    stream_time -= MPI_Wtime();
    R reducer = 0;
    size_t basic_chunk = 64;
    for (int t_i=0;t_i<threads;t_i++) {
      int s_i = get_socket_id(t_i);
      int s_j = get_socket_offset(t_i);
      VertexId curr_offset_beg = local_partition_offset[s_i]>=partition_offset[partition_id] ? local_partition_offset[s_i] : partition_offset[partition_id];
      VertexId curr_offset_end = local_partition_offset[s_i+1]<=partition_offset[partition_id+1] ? local_partition_offset[s_i+1] : partition_offset[partition_id+1];
      VertexId partition_size = curr_offset_end - curr_offset_beg;
      thread_state[t_i]->curr = curr_offset_beg + partition_size / threads_per_socket  / basic_chunk * basic_chunk * s_j;
      thread_state[t_i]->end = curr_offset_beg + partition_size / threads_per_socket / basic_chunk * basic_chunk * (s_j+1);


      if (s_j == threads_per_socket - 1) {
        thread_state[t_i]->end = curr_offset_end;
      }
      thread_state[t_i]->status = WORKING;
    }
    #pragma omp parallel reduction(+:reducer)
    {
      R local_reducer = 0;
      int thread_id = omp_get_thread_num();
      while (true) {
        VertexId v_i = __sync_fetch_and_add(&thread_state[thread_id]->curr, basic_chunk);
        if (v_i >= thread_state[thread_id]->end || v_i >= partition_overlap_offset[end_id]) break;
        else if (v_i >= vertices) v_i = v_i-vertices;
        else if (v_i <0) v_i = v_i+vertices;
        unsigned long word = active->data[WORD_OFFSET(v_i)];
        while (word != 0) {
          if (word & 1) {
            local_reducer += process(v_i);
          }
          v_i++;
          word = word >> 1;
        }
      }
      thread_state[thread_id]->status = STEALING;
      for (int t_offset=1;t_offset<threads;t_offset++) {
        int t_i = (thread_id + t_offset) % threads;
        while (thread_state[t_i]->status!=STEALING) {
          VertexId v_i = __sync_fetch_and_add(&thread_state[t_i]->curr, basic_chunk);
          if (v_i >= thread_state[t_i]->end||v_i >= partition_overlap_offset[end_id]) continue;
          else if (v_i >= vertices) v_i = v_i-vertices;
          else if (v_i <0) v_i = v_i+vertices;
          unsigned long word = active->data[WORD_OFFSET(v_i)];
          while (word != 0) {
            if (word & 1) {
              local_reducer += process(v_i);
            }
            v_i++;
            word = word >> 1;
          }
        }
      }
      reducer += local_reducer;
    }
    R global_reducer;
    MPI_Datatype dt = get_mpi_data_type<R>();
    MPI_Allreduce(&reducer, &global_reducer, 1, dt, MPI_SUM, MPI_COMM_WORLD);
    stream_time += MPI_Wtime();
    #ifdef PRINT_DEBUG_MESSAGES
    if (partition_id==0) {
      printf("process_vertices took %lf (s)\n", stream_time);
    }
    #endif
    return global_reducer;
  }

  template<typename M>
  void flush_local_send_buffer(int t_i) {
    int s_i = get_socket_id(t_i);
    int pos = __sync_fetch_and_add(&send_buffer[current_send_part_id][s_i]->count, local_send_buffer[t_i]->count);
    memcpy(send_buffer[current_send_part_id][s_i]->data + sizeof(MsgUnit<M>) * pos, local_send_buffer[t_i]->data, sizeof(MsgUnit<M>) * local_send_buffer[t_i]->count);
    local_send_buffer[t_i]->count = 0;
  }

  // emit a message to a vertex's master (dense) / mirror (sparse)
  template<typename M>
  void emit(VertexId vtx, M msg) {
    int t_i = omp_get_thread_num();
    MsgUnit<M> * buffer = (MsgUnit<M>*)local_send_buffer[t_i]->data;
    buffer[local_send_buffer[t_i]->count].vertex = vtx;
    buffer[local_send_buffer[t_i]->count].msg_data = msg;
    local_send_buffer[t_i]->count += 1;
    if (local_send_buffer[t_i]->count==local_send_buffer_limit) {
      flush_local_send_buffer<M>(t_i);
    }
  }

  // process edges
  template<typename R, typename M>
  R process_edges(std::function<void(VertexId)> sparse_signal, std::function<R(VertexId, M, VertexAdjList<EdgeData>)> sparse_slot, std::function<void(VertexId,  VertexAdjList<EdgeData>)> dense_signal, std::function<R(VertexId, M)> dense_slot, Bitmap * active, Bitmap * dense_selective = nullptr, int src_only =0 ) {
    double stream_time = 0;
    stream_time -= MPI_Wtime();

    for (int t_i=0;t_i<threads;t_i++) {
      local_send_buffer[t_i]->resize( sizeof(MsgUnit<M>) * local_send_buffer_limit );
      local_send_buffer[t_i]->count = 0;
    }
    R reducer = 0;
    EdgeId active_edges = process_vertices<EdgeId>(
      [&](VertexId vtx){
        return (EdgeId)out_degree[vtx];
      },
      active
    );
    //printf("%d:%llu\n",partition_id,active_edges);
    bool sparse = (active_edges < edges / 20);

    if(src_only)
      sparse = false;
     
    if (sparse) {
      for (int i=0;i<partitions;i++) {
        for (int s_i=0;s_i<sockets;s_i++) {
          recv_buffer[i][s_i]->resize( sizeof(MsgUnit<M>) * (partition_offset[i+1] - partition_offset[i]) * sockets );
          send_buffer[i][s_i]->resize( sizeof(MsgUnit<M>) * (partition_offset[partition_id+1]-partition_offset[partition_id]) * sockets );
          send_buffer[i][s_i]->count = 0;
          recv_buffer[i][s_i]->count = 0;
        }
      }
    } else {
      for (int i=0;i<partitions;i++) {
        for (int s_i=0;s_i<sockets;s_i++) {
          recv_buffer[i][s_i]->resize( sizeof(MsgUnit<M>) * (partition_offset[partition_id+1]-partition_offset[partition_id]) * sockets );
          send_buffer[i][s_i]->resize( sizeof(MsgUnit<M>) * (partition_offset[i+1] - partition_offset[i]) * sockets );
          send_buffer[i][s_i]->count = 0;
          recv_buffer[i][s_i]->count = 0;
        }
      }
    }
    size_t basic_chunk = 64;
    if (sparse) {
      #ifdef PRINT_DEBUG_MESSAGES
      if (partition_id==0) {
        printf("sparse mode\n");
      }
      #endif
      int * recv_queue = new int [partitions];
      int recv_queue_size = 0;
      std::mutex recv_queue_mutex;

      current_send_part_id = partition_id;
      #pragma omp parallel for
      for (VertexId begin_v_i=partition_offset[partition_id];begin_v_i<partition_offset[partition_id+1];begin_v_i+=basic_chunk) {
        VertexId v_i = begin_v_i < 0 ? begin_v_i + vertices :
                       begin_v_i >= vertices ? begin_v_i - vertices : begin_v_i;
        unsigned long word = active->data[WORD_OFFSET(v_i)];
        while (word != 0) {
          if (word & 1) {
            sparse_signal(v_i);
          }
          v_i++;
          word = word >> 1;
        }
      }
      #pragma omp parallel for
      for (int t_i=0;t_i<threads;t_i++) {
        flush_local_send_buffer<M>(t_i);
      }
      recv_queue[recv_queue_size] = partition_id;
      recv_queue_mutex.lock();
      recv_queue_size += 1;
      recv_queue_mutex.unlock();
      std::thread send_thread([&](){
        for (int step=1;step<partitions;step++) {
          //comm time
          unsigned long long begin_comm_time = rdtscl();
          int i = (partition_id - step + partitions) % partitions;
          for (int s_i=0;s_i<sockets;s_i++) {
            MPI_Send(send_buffer[partition_id][s_i]->data, sizeof(MsgUnit<M>) * send_buffer[partition_id][s_i]->count, MPI_CHAR, i, PassMessage, MPI_COMM_WORLD);
          }
          comm_time[i] = rdtscl() - begin_comm_time;
          #ifdef PRINT_DEBUG_MESSAGES
          printf("node%d costs %llu to send buffer to node %d\n",partition_id,comm_time[i],i);
          #endif
        }
      });
      std::thread recv_thread([&](){
        for (int step=1;step<partitions;step++) {
          int i = (partition_id + step) % partitions;
          for (int s_i=0;s_i<sockets;s_i++) {
            MPI_Status recv_status;
            MPI_Probe(i, PassMessage, MPI_COMM_WORLD, &recv_status);
            MPI_Get_count(&recv_status, MPI_CHAR, &recv_buffer[i][s_i]->count);
            MPI_Recv(recv_buffer[i][s_i]->data, recv_buffer[i][s_i]->count, MPI_CHAR, i, PassMessage, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            recv_buffer[i][s_i]->count /= sizeof(MsgUnit<M>);
          }
          recv_queue[recv_queue_size] = i;
          recv_queue_mutex.lock();
          recv_queue_size += 1;
          recv_queue_mutex.unlock();
        }
      });
      //precise time
      unsigned long long begin_rdtscl = rdtscl();
      for (int step=0;step<partitions;step++) {
        while (true) {
          recv_queue_mutex.lock();
          bool condition = (recv_queue_size<=step);
          recv_queue_mutex.unlock();
          if (!condition) break;
          __asm volatile ("pause" ::: "memory");
        }
        int i = recv_queue[step];
        MessageBuffer ** used_buffer;
        if (i==partition_id) {
          used_buffer = send_buffer[i];
        } else {
          used_buffer = recv_buffer[i];
        }
        for (int s_i=0;s_i<sockets;s_i++) {
          MsgUnit<M> * buffer = (MsgUnit<M> *)used_buffer[s_i]->data;
          size_t buffer_size = used_buffer[s_i]->count;
          for (int t_i=0;t_i<threads;t_i++) {
            // int s_i = get_socket_id(t_i);
            int s_j = get_socket_offset(t_i);
            VertexId partition_size = buffer_size;
            thread_state[t_i]->curr = partition_size / threads_per_socket  / basic_chunk * basic_chunk * s_j;
            thread_state[t_i]->end = partition_size / threads_per_socket / basic_chunk * basic_chunk * (s_j+1);
            if (s_j == threads_per_socket - 1) {
              thread_state[t_i]->end = buffer_size;
            }
            thread_state[t_i]->status = WORKING;
          }
          #pragma omp parallel reduction(+:reducer)
          {
            R local_reducer = 0;
            int thread_id = omp_get_thread_num();
            int s_i = get_socket_id(thread_id);
            while (true) {
              VertexId b_i = __sync_fetch_and_add(&thread_state[thread_id]->curr, basic_chunk);
              if (b_i >= thread_state[thread_id]->end) break;
              VertexId begin_b_i = b_i;
              VertexId end_b_i = b_i + basic_chunk;
              if (end_b_i>thread_state[thread_id]->end) {
                end_b_i = thread_state[thread_id]->end;
              }
              for (b_i=begin_b_i;b_i<end_b_i;b_i++) {
                VertexId v_i = buffer[b_i].vertex;
                M msg_data = buffer[b_i].msg_data;
                if (outgoing_adj_bitmap[s_i]->get_bit(v_i)) {
                  if(partition_offset[partition_id]>=0 && partition_offset[partition_id+1]<=vertices)
                    local_reducer += sparse_slot(v_i, msg_data, VertexAdjList<EdgeData>(outgoing_adj_list[s_i] + outgoing_adj_index[s_i][v_i], outgoing_adj_list[s_i] + outgoing_adj_index[s_i][v_i+1],partition_offset[partition_id],partition_offset[partition_id+1],vertices));
                  else{
                    local_reducer += sparse_slot(v_i, msg_data, VertexAdjList<EdgeData>(outgoing_adj_list[s_i] + outgoing_adj_index[s_i][v_i], outgoing_adj_list[s_i] + outgoing_adj_index[s_i][v_i+1],partition_offset[partition_id],vertices,vertices));
                    local_reducer += sparse_slot(v_i, msg_data, VertexAdjList<EdgeData>(outgoing_adj_list[s_i] + outgoing_adj_index[s_i][v_i], outgoing_adj_list[s_i] + outgoing_adj_index[s_i][v_i+1],0,partition_offset[partition_id+1],vertices));
                  }
                }
              }
            }
            thread_state[thread_id]->status = STEALING;
            for (int t_offset=1;t_offset<threads;t_offset++) {
              int t_i = (thread_id + t_offset) % threads;
              if (thread_state[t_i]->status==STEALING) continue;
              while (true) {
                VertexId b_i = __sync_fetch_and_add(&thread_state[t_i]->curr, basic_chunk);
                if (b_i >= thread_state[t_i]->end) break;
                VertexId begin_b_i = b_i;
                VertexId end_b_i = b_i + basic_chunk;
                if (end_b_i>thread_state[t_i]->end) {
                  end_b_i = thread_state[t_i]->end;
                }
                int s_i = get_socket_id(t_i);
                for (b_i=begin_b_i;b_i<end_b_i;b_i++) {
                  VertexId v_i = buffer[b_i].vertex;
                  M msg_data = buffer[b_i].msg_data;
                  if (outgoing_adj_bitmap[s_i]->get_bit(v_i)) {
                    if(partition_offset[partition_id]>=0 && partition_offset[partition_id+1]<=vertices)
                      local_reducer += sparse_slot(v_i, msg_data, VertexAdjList<EdgeData>(outgoing_adj_list[s_i] + outgoing_adj_index[s_i][v_i], outgoing_adj_list[s_i] + outgoing_adj_index[s_i][v_i+1],partition_offset[partition_id],partition_offset[partition_id+1],vertices));
                    else{
                      local_reducer += sparse_slot(v_i, msg_data, VertexAdjList<EdgeData>(outgoing_adj_list[s_i] + outgoing_adj_index[s_i][v_i], outgoing_adj_list[s_i] + outgoing_adj_index[s_i][v_i+1],partition_offset[partition_id],vertices,vertices));
                      local_reducer += sparse_slot(v_i, msg_data, VertexAdjList<EdgeData>(outgoing_adj_list[s_i] + outgoing_adj_index[s_i][v_i], outgoing_adj_list[s_i] + outgoing_adj_index[s_i][v_i+1],0,partition_offset[partition_id+1],vertices));
                    }
                  }
                }
              }
            }
            reducer += local_reducer;
          }
        }
        unsigned long long span_rdtscl = rdtscl() - begin_rdtscl;
        comm_time[partition_id] = span_rdtscl;
      }
      #ifdef PRINT_DEBUG_MESSAGES
      printf("node%d costs %llu to process\n",partition_id,comm_time[partition_id]);
      #endif
      send_thread.join();
      recv_thread.join();
      delete [] recv_queue;
    } else {
      // dense selective bitmap
      if (dense_selective!=nullptr && partitions>1) {
        double sync_time = 0;
        sync_time -= get_time();
        std::thread send_thread([&](){
          for (int step=1;step<partitions;step++) {
            int recipient_id = (partition_id + step) % partitions;
            MPI_Send(dense_selective->data + WORD_OFFSET(partition_offset[partition_id]), (partition_offset[partition_id+1]-partition_offset[partition_id]) / 64, MPI_UNSIGNED_LONG, recipient_id, PassMessage, MPI_COMM_WORLD);
          }
        });
        std::thread recv_thread([&](){
          for (int step=1;step<partitions;step++) {
            int sender_id = (partition_id - step + partitions) % partitions;
            MPI_Recv(dense_selective->data + WORD_OFFSET(partition_offset[sender_id]), (partition_offset[sender_id + 1] - partition_offset[sender_id]) / 64, MPI_UNSIGNED_LONG, sender_id, PassMessage, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
          }
        });
        send_thread.join();
        recv_thread.join();
        MPI_Barrier(MPI_COMM_WORLD);
        sync_time += get_time();
        #ifdef PRINT_DEBUG_MESSAGES
        if (partition_id==0) {
          printf("sync_time = %lf\n", sync_time);
        }
        #endif
      }

      if (src_only && partitions>1) {
        double sync_time = 0;
        sync_time -= get_time();
        std::thread send_thread([&](){
          for (int step=1;step<partitions;step++) {
            int recipient_id = (partition_id + step) % partitions;
            MPI_Send(active->data + WORD_OFFSET(partition_offset[partition_id]), (partition_offset[partition_id+1]-partition_offset[partition_id]) / 64, MPI_UNSIGNED_LONG, recipient_id, PassMessage, MPI_COMM_WORLD);
          }
        });
        std::thread recv_thread([&](){
          for (int step=1;step<partitions;step++) {
            int sender_id = (partition_id - step + partitions) % partitions;
            MPI_Recv(active->data + WORD_OFFSET(partition_offset[sender_id]), (partition_offset[sender_id + 1] - partition_offset[sender_id]) / 64, MPI_UNSIGNED_LONG, sender_id, PassMessage, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
          }
        });
        send_thread.join();
        recv_thread.join();
        MPI_Barrier(MPI_COMM_WORLD);
        sync_time += get_time();
        #ifdef PRINT_DEBUG_MESSAGES
        if (partition_id==0) {
          printf("sync_time = %lf\n", sync_time);
        }                                                                                                                 
        #endif                                                                                                            
      } 

      #ifdef PRINT_DEBUG_MESSAGES
      if (partition_id==0) {
        printf("dense mode\n");
      }
      #endif
      int * send_queue = new int [partitions];
      int * recv_queue = new int [partitions];
      volatile int send_queue_size = 0;
      volatile int recv_queue_size = 0;
      std::mutex send_queue_mutex;
      std::mutex recv_queue_mutex;

      std::thread send_thread([&](){
        for (int step=0;step<partitions;step++) {
          unsigned long long begin_comm_time = rdtscl();
          if (step==partitions-1) {
            break;
          }
          while (true) {
            send_queue_mutex.lock();
            bool condition = (send_queue_size<=step);
            send_queue_mutex.unlock();
            if (!condition) break;
            __asm volatile ("pause" ::: "memory");
          }
          //start_send_time[partition_id][send_queue[step]] = time()
          int i = send_queue[step];
          for (int s_i=0;s_i<sockets;s_i++) {
            MPI_Send(send_buffer[i][s_i]->data, sizeof(MsgUnit<M>) * send_buffer[i][s_i]->count, MPI_CHAR, i, PassMessage, MPI_COMM_WORLD);
          }
          comm_time[i] = rdtscl() - begin_comm_time;
          #ifdef PRINT_DEBUG_MESSAGES
          printf("node%d costs %llu to send buffer to node %d\n",partition_id,comm_time[i],i);
          #endif
        }
      });
      std::thread recv_thread([&](){
        std::vector<std::thread> threads;
        for (int step=1;step<partitions;step++) {
          int i = (partition_id - step + partitions) % partitions;
          threads.emplace_back([&](int i){
            for (int s_i=0;s_i<sockets;s_i++) {
              MPI_Status recv_status;
              MPI_Probe(i, PassMessage, MPI_COMM_WORLD, &recv_status);
              MPI_Get_count(&recv_status, MPI_CHAR, &recv_buffer[i][s_i]->count);
              MPI_Recv(recv_buffer[i][s_i]->data, recv_buffer[i][s_i]->count, MPI_CHAR, i, PassMessage, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
              recv_buffer[i][s_i]->count /= sizeof(MsgUnit<M>);
            }
          }, i);
        }
        for (int step=1;step<partitions;step++) {
          int i = (partition_id - step + partitions) % partitions;
          threads[step-1].join();
          recv_queue[recv_queue_size] = i;
          recv_queue_mutex.lock();
          recv_queue_size += 1;
          recv_queue_mutex.unlock();
        }
        recv_queue[recv_queue_size] = partition_id;
        recv_queue_mutex.lock();
        recv_queue_size += 1;
        recv_queue_mutex.unlock();
      });
      current_send_part_id = partition_id;
      unsigned long long begin_time = rdtscl();
      for (int step=0;step<partitions;step++) {
        current_send_part_id = (current_send_part_id + 1) % partitions;
        int i = current_send_part_id;
        compute_time[i] = rdtscl();
        for (int t_i=0;t_i<threads;t_i++) {
          *thread_state[t_i] = tuned_chunks_dense[i][t_i];
        }
        #pragma omp parallel
        {
          int thread_id = omp_get_thread_num();
          int s_i = get_socket_id(thread_id);
          VertexId final_p_v_i = thread_state[thread_id]->end;
          while (true) {
            VertexId begin_p_v_i = __sync_fetch_and_add(&thread_state[thread_id]->curr, basic_chunk);
            if (begin_p_v_i >= final_p_v_i) break;
            VertexId end_p_v_i = begin_p_v_i + basic_chunk;
            if (end_p_v_i > final_p_v_i) {
              end_p_v_i = final_p_v_i;
            }
            for (VertexId p_v_i = begin_p_v_i; p_v_i < end_p_v_i; p_v_i ++) {
              VertexId v_i = compressed_incoming_adj_index[s_i][p_v_i].vertex;
              dense_signal(v_i, VertexAdjList<EdgeData>(incoming_adj_list[s_i] + compressed_incoming_adj_index[s_i][p_v_i].index, incoming_adj_list[s_i] + compressed_incoming_adj_index[s_i][p_v_i+1].index));
            }
          }
          thread_state[thread_id]->status = STEALING;
          for (int t_offset=1;t_offset<threads;t_offset++) {
            int t_i = (thread_id + t_offset) % threads;
            int s_i = get_socket_id(t_i);
            while (thread_state[t_i]->status!=STEALING) {
              VertexId begin_p_v_i = __sync_fetch_and_add(&thread_state[t_i]->curr, basic_chunk);
              if (begin_p_v_i >= thread_state[t_i]->end) break;
              VertexId end_p_v_i = begin_p_v_i + basic_chunk;
              if (end_p_v_i > thread_state[t_i]->end) {
                end_p_v_i = thread_state[t_i]->end;
              }
              for (VertexId p_v_i = begin_p_v_i; p_v_i < end_p_v_i; p_v_i ++) {
                VertexId v_i = compressed_incoming_adj_index[s_i][p_v_i].vertex;
                dense_signal(v_i, VertexAdjList<EdgeData>(incoming_adj_list[s_i] + compressed_incoming_adj_index[s_i][p_v_i].index, incoming_adj_list[s_i] + compressed_incoming_adj_index[s_i][p_v_i+1].index));
              }
            }
          }
        }
        #pragma omp parallel for
        for (int t_i=0;t_i<threads;t_i++) {
          flush_local_send_buffer<M>(t_i);
        }
        if (i!=partition_id) {
          send_queue[send_queue_size] = i;
          send_queue_mutex.lock();
          send_queue_size += 1;
          send_queue_mutex.unlock();
        }
        //process time
        unsigned long long span_time = rdtscl() - begin_time;
        comm_time[partition_id] = span_time;
        compute_time[i] = rdtscl() - compute_time[i];
      }
      
      printf("node%d costs %llu %llu %llu %llu\n",partition_id,compute_time[0],compute_time[1],compute_time[2],compute_time[3]);


      for (int step=0;step<partitions;step++) {
        //nowtime = time()
        //unsigned long long nowtime = rdtscl();
        while (true) {
          recv_queue_mutex.lock();
          bool condition = (recv_queue_size<=step);
          recv_queue_mutex.unlock();
          if (!condition) break;
          __asm volatile ("pause" ::: "memory");
        }
        //recv_queue[step] reaches partition_id in there
        //nowtime2 = time()
        //waiting_time[recv_queue[step]][partition_id] = nowtime - start_send_time[recv_queue[step]][partition_id]
        //printf("node %d recv %d cost %llu\n",partition_id,recv_queue[step],rdtscl()-nowtime);
        int i = recv_queue[step];
        MessageBuffer ** used_buffer;
        if (i==partition_id) {
          used_buffer = send_buffer[i];
        } else {
          used_buffer = recv_buffer[i];
        }
        for (int t_i=0;t_i<threads;t_i++) {
          int s_i = get_socket_id(t_i);
          int s_j = get_socket_offset(t_i);
          VertexId partition_size = used_buffer[s_i]->count;
          thread_state[t_i]->curr = partition_size / threads_per_socket  / basic_chunk * basic_chunk * s_j;
          thread_state[t_i]->end = partition_size / threads_per_socket / basic_chunk * basic_chunk * (s_j+1);
          if (s_j == threads_per_socket - 1) {
            thread_state[t_i]->end = used_buffer[s_i]->count;
          }
          thread_state[t_i]->status = WORKING;
        }
        #pragma omp parallel reduction(+:reducer)
        {
          R local_reducer = 0;
          int thread_id = omp_get_thread_num();
          int s_i = get_socket_id(thread_id);
          MsgUnit<M> * buffer = (MsgUnit<M> *)used_buffer[s_i]->data;
          while (true) {
            VertexId b_i = __sync_fetch_and_add(&thread_state[thread_id]->curr, basic_chunk);
            if (b_i >= thread_state[thread_id]->end) break;
            VertexId begin_b_i = b_i;
            VertexId end_b_i = b_i + basic_chunk;
            if (end_b_i>thread_state[thread_id]->end) {
              end_b_i = thread_state[thread_id]->end;
            }
            for (b_i=begin_b_i;b_i<end_b_i;b_i++) {
              VertexId v_i = buffer[b_i].vertex;
              M msg_data = buffer[b_i].msg_data;
              local_reducer += dense_slot(v_i, msg_data);
            }
          }
          thread_state[thread_id]->status = STEALING;
          reducer += local_reducer;
        }
      }
      send_thread.join();
      recv_thread.join();
      delete [] send_queue;
      delete [] recv_queue;
    }

    R global_reducer;
    MPI_Datatype dt = get_mpi_data_type<R>();
    MPI_Allreduce(&reducer, &global_reducer, 1, dt, MPI_SUM, MPI_COMM_WORLD);
    stream_time += MPI_Wtime();
    #ifdef PRINT_DEBUG_MESSAGES
    if (partition_id==0) {
      printf("process_edges took %lf (s)\n", stream_time);
    }
    #endif
    return global_reducer;
  }

  void broadcast_commtimes(){
    if(partition_id!=0){
      MPI_Send(compute_time,partitions,MPI_UNSIGNED_LONG_LONG,0,Schedule,MPI_COMM_WORLD);
    }
    else{
      memcpy(timekeeper[0],compute_time,sizeof(unsigned long long)*partitions);
      for(int i=1;i<partitions;i++)
        MPI_Recv(timekeeper[i],partitions,MPI_UNSIGNED_LONG_LONG,i,Schedule,MPI_COMM_WORLD,MPI_STATUS_IGNORE);
    }
    MPI_Barrier(MPI_COMM_WORLD);
  }

  void schedule(){
    broadcast_commtimes();
    if(partition_id==0){
      unsigned long long * means = new unsigned long long[partitions];
      unsigned long long mean = 0;
      int * steps = new int [partitions-1];
      for(int i=0;i<partitions;i++){
        means[i]=0;
        for(int j=0;j<partitions;j++)
          means[i]+=timekeeper[j][i];
        means[i]/=partitions;
        mean+=means[i];
      }
      mean/=partitions;
      for(int i=0;i<partitions-1;i++){
        steps[i]=int((1.05-float(means[i])/float(mean))*10);
        //printf("%d %f",steps[i],float(means[i])/float(mean));
      }
      //printf("\n");
      int sum=0;
      for(int j=0;j<partitions-1;j++){
        int tag=0;
        if(steppointer[j+1]+sum+steps[j]<=20&&steppointer[j+1]+sum+steps[j]>=0)
            sum+=steps[j];
        else if(steppointer[j+1]+sum+steps[j]>20){
            steps[j]=20-(steppointer[j+1]+sum);
            sum+=steps[j];
        }else{
            steps[j]=0-(steppointer[j+1]+sum);
            sum+=steps[j];
        }
        new_partition_offset[j+1]=partition_offset[j+1];
        if(sum<0){
          for(int k=0;k>sum;k--){
            if(steppointer[j+1]==11)
              new_partition_offset[j+1]=new_partition_offset[j+1]-stepsize[j+1][2];
            else if(steppointer[j+1]==10)
              new_partition_offset[j+1]=new_partition_offset[j+1]-stepsize[j+1][1];
            else if(steppointer[j+1]>11)
              new_partition_offset[j+1]=new_partition_offset[j+1]-stepsize[j+1][3];
            else if(steppointer[j+1]!=0)
              new_partition_offset[j+1]=new_partition_offset[j+1]-stepsize[j+1][0];
            else
              new_partition_offset[j+1]=new_partition_offset[j+1];
            if(steppointer[j+1]!=0)steppointer[j+1]--;
          }
        }else if(sum>0){
          for(int k=0;k<sum;k++){
            if(steppointer[j+1]==9)
              new_partition_offset[j+1]=new_partition_offset[j+1]+stepsize[j+1][1];
            else if(steppointer[j+1]==10)
              new_partition_offset[j+1]=new_partition_offset[j+1]+stepsize[j+1][2];
            else if(steppointer[j+1]<9)
              new_partition_offset[j+1]=new_partition_offset[j+1]+stepsize[j+1][0];
            else if(steppointer[j+1]!=20)
              new_partition_offset[j+1]=new_partition_offset[j+1]+stepsize[j+1][3];
            else
              new_partition_offset[j+1]=new_partition_offset[j+1];
            if(steppointer[j+1]!=20)steppointer[j+1]++;
          }
        }else
          new_partition_offset[j+1]=partition_offset[j+1];
      }
    }
    MPI_Bcast(steppointer,partitions+1,MPI_INT,0,MPI_COMM_WORLD);
    MPI_Bcast(new_partition_offset,partitions+1,MPI_INT,0,MPI_COMM_WORLD);
    for(int j=0;j<partitions-1;j++){
      for(int t_i=0;t_i<threads;t_i++) {
          int s_i = get_socket_id(t_i);
          int s_j = get_socket_offset(t_i);
          if(s_j==0){
            tuned_chunks_dense[j+1][t_i].curr=tuned_chunks_currs[j+1][s_i][steppointer[j+1]];
          }
          if(s_j==threads_per_socket-1)
            tuned_chunks_dense[j][t_i].end=tuned_chunks_ends[j][s_i][steppointer[j+1]];
      }
    }

    /*if(steppointer[0]==10){
      new_partition_offset[0]=partition_offset[0]-stepsize[0][1];
      new_partition_offset[partitions]=partition_offset[partitions]-stepsize[0][1];
      if(new_partition_offset[0] != new_partition_offset[partitions] - vertices)printf("not same\n");
      steppointer[0]=9;
    }*/

    new_partition_offset[partitions] = partition_offset[partitions];
    new_partition_offset[0] = partition_offset[0];
    //new_partition_offset[0] = new_partition_offset[partitions] - vertices;

    return;
  }

  template<typename T>
  void sync_arr_new(T * array){
    if(partition_id==0)
      for(int j=0;j<=partitions;j++)
        printf("##offset %d %d\n",partition_offset[j],new_partition_offset[j]);

    VertexArray * p = vertex_head.headaddr;
    VertexArray * r = p->next;

    while(reinterpret_cast<T*>(p->add)!=array){
      p=r;
      r=p->next;
    }

      std::thread right_send_thread([&](){
        int recipient_id = (partition_id + 1) % partitions;
        if(partition_id != partitions - 1){
          if(partition_offset[recipient_id] > new_partition_offset[recipient_id]){
            MPI_Send(p->add+new_partition_offset[recipient_id]*p->type_size, (partition_offset[recipient_id]-new_partition_offset[recipient_id])*(p->type_size), MPI_CHAR, recipient_id, GatherVertexArray, MPI_COMM_WORLD);
          }
        }else{
          if(partition_offset[recipient_id] > new_partition_offset[recipient_id]){
            if(partition_offset[recipient_id] <= 0 || new_partition_offset[recipient_id] >= 0){
              MPI_Send(p->add+(vertices+new_partition_offset[recipient_id])%vertices*p->type_size, (partition_offset[recipient_id]-new_partition_offset[recipient_id])*(p->type_size), MPI_CHAR, recipient_id, GatherVertexArray, MPI_COMM_WORLD);
            }else{
              MPI_Send(p->add+(vertices+new_partition_offset[recipient_id])*p->type_size, (0-new_partition_offset[recipient_id])*(p->type_size), MPI_CHAR, recipient_id, GatherVertexArray, MPI_COMM_WORLD);
              MPI_Send(p->add, partition_offset[recipient_id]*(p->type_size), MPI_CHAR, recipient_id, GatherVertexArray, MPI_COMM_WORLD);
            }
          }
        }
      });

      std::thread left_recv_thread([&](){
        int sender_id = (partition_id - 1 + partitions) % partitions;
        if(partition_id != 0){
          if(partition_offset[partition_id] > new_partition_offset[partition_id]){
            MPI_Recv(p->add+new_partition_offset[partition_id]*p->type_size, (partition_offset[partition_id]-new_partition_offset[partition_id])*(p->type_size), MPI_CHAR, sender_id, GatherVertexArray,MPI_COMM_WORLD, MPI_STATUS_IGNORE);
          }
        }else{
          if(partition_offset[partitions] > new_partition_offset[partitions]){
            if(partition_offset[partitions] <= vertices || new_partition_offset[partitions] >= vertices){
              MPI_Recv(p->add+new_partition_offset[partitions]%vertices*p->type_size, (partition_offset[partitions]-new_partition_offset[partitions])*(p->type_size), MPI_CHAR, sender_id, GatherVertexArray, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            }else{
              MPI_Recv(p->add+(new_partition_offset[partitions])*p->type_size, (vertices-new_partition_offset[partitions])*(p->type_size), MPI_CHAR, sender_id, GatherVertexArray, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
              MPI_Recv(p->add, (partition_offset[partitions]-vertices)*(p->type_size), MPI_CHAR, sender_id, GatherVertexArray, MPI_COMM_WORLD,MPI_STATUS_IGNORE);
            }
          }
        }
      });

      std::thread left_send_thread([&](){
        int recipient_id = (partition_id - 1 + partitions) % partitions;
        if(partition_id != 0){
          if(partition_offset[partition_id] < new_partition_offset[partition_id]){
            MPI_Send(p->add+partition_offset[partition_id]*p->type_size, (new_partition_offset[partition_id]-partition_offset[partition_id])*(p->type_size), MPI_CHAR, recipient_id, GatherVertexArray, MPI_COMM_WORLD);
          }
        }else{
          if(partition_offset[partition_id] < new_partition_offset[partition_id]){
            if(partition_offset[partition_id] >= 0 || new_partition_offset[partition_id] <= 0){
              MPI_Send(p->add+(vertices+partition_offset[partition_id])%vertices*p->type_size, (new_partition_offset[partition_id]-partition_offset[partition_id])*(p->type_size), MPI_CHAR, recipient_id, GatherVertexArray, MPI_COMM_WORLD);
            }else{
              MPI_Send(p->add+(vertices+partition_offset[partition_id])%vertices*p->type_size, (0-partition_offset[partition_id])*(p->type_size), MPI_CHAR, recipient_id, GatherVertexArray, MPI_COMM_WORLD);
              MPI_Send(p->add, new_partition_offset[partition_id]%vertices*(p->type_size), MPI_CHAR, recipient_id, GatherVertexArray, MPI_COMM_WORLD); // % vertices may not be necesssary
            }
          }
        }
      });

      std::thread right_recv_thread([&](){
        int sender_id = (partition_id + 1) % partitions;
        if(partition_id != partitions - 1){
          if(partition_offset[sender_id] < new_partition_offset[sender_id]){
            MPI_Recv(p->add+partition_offset[sender_id]*p->type_size, (new_partition_offset[sender_id]-partition_offset[sender_id])*(p->type_size), MPI_CHAR, sender_id, GatherVertexArray,MPI_COMM_WORLD, MPI_STATUS_IGNORE);
          }
        }else{
          if(partition_offset[sender_id] < new_partition_offset[sender_id]){
            if(partition_offset[sender_id] >= 0 || new_partition_offset[sender_id] <= 0){
              MPI_Recv(p->add+(vertices+partition_offset[sender_id])%vertices*p->type_size, (new_partition_offset[sender_id]-partition_offset[sender_id])*(p->type_size), MPI_CHAR, sender_id, GatherVertexArray, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            }else{
              MPI_Recv(p->add+(vertices+partition_offset[sender_id])*p->type_size, (0-partition_offset[sender_id])*(p->type_size), MPI_CHAR, sender_id, GatherVertexArray, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
              MPI_Recv(p->add, new_partition_offset[sender_id]*(p->type_size), MPI_CHAR, sender_id, GatherVertexArray, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            }
          }
        }
      });

      right_send_thread.join();
      left_recv_thread.join();
      left_send_thread.join();
      right_recv_thread.join();
      MPI_Barrier(MPI_COMM_WORLD);

    for(int i=0;i<=partitions;i++)
      partition_offset[i]=new_partition_offset[i];
  }

  template<typename T>
  void sync_arr(T * array){
    if(partition_id==0)
      for(int j=0;j<=partitions;j++)
        printf("##offset %d %d\n",partition_offset[j],new_partition_offset[j]);
    
    VertexArray * p = vertex_head.headaddr;
    VertexArray * r = p->next;

    while(reinterpret_cast<T*>(p->add)!=array){
      p=r;
      r=p->next;
    }

    std::thread right_thread([&](){
      if(partition_id!=partitions-1){
        int recipient_id = (partition_id + 1) % partitions;
        if(partition_offset[recipient_id]>new_partition_offset[recipient_id]){
          MPI_Send(p->add+new_partition_offset[recipient_id]*p->type_size, (partition_offset[recipient_id]-new_partition_offset[recipient_id])*(p->type_size), MPI_CHAR, recipient_id, GatherVertexArray, MPI_COMM_WORLD);
        }
        else if(partition_offset[recipient_id]<new_partition_offset[recipient_id])
          MPI_Recv(p->add+partition_offset[recipient_id]*p->type_size, (new_partition_offset[recipient_id]-partition_offset[recipient_id])*(p->type_size), MPI_CHAR, recipient_id, GatherVertexArray,MPI_COMM_WORLD, MPI_STATUS_IGNORE);
      }else{//from the last node to the 1st node
        if(partition_offset[0]>new_partition_offset[0]){//send mode
          if(partition_offset[0]<=0 || new_partition_offset[0]>=0){
            MPI_Send(p->add+(vertices+new_partition_offset[0])%vertices*p->type_size, (partition_offset[0]-new_partition_offset[0])*(p->type_size), MPI_CHAR, 0, GatherVertexArray, MPI_COMM_WORLD);
         }
           else{
            MPI_Send(p->add+(vertices+new_partition_offset[0])*p->type_size, (0-new_partition_offset[0])*(p->type_size), MPI_CHAR, 0, GatherVertexArray, MPI_COMM_WORLD);
            MPI_Send(p->add, partition_offset[0]*(p->type_size), MPI_CHAR, 0, GatherVertexArray, MPI_COMM_WORLD);
           }
         }else if(partition_offset[0]<new_partition_offset[0]){//recv mode
           if(partition_offset[0]>=0 || new_partition_offset[0]<=0)
             MPI_Recv(p->add+(vertices+partition_offset[0])%vertices*p->type_size, (new_partition_offset[0]-partition_offset[0])*(p->type_size), MPI_CHAR, 0, GatherVertexArray, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
           else{
             MPI_Recv(p->add+(vertices+partition_offset[0])*p->type_size, (0-partition_offset[0])*(p->type_size), MPI_CHAR, 0, GatherVertexArray, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
             MPI_Recv(p->add, new_partition_offset[0]*(p->type_size), MPI_CHAR, 0, GatherVertexArray, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
           }
         }//recv mode
      }//from the last node to the 1st node
    });
    std::thread left_thread([&](){
      if(partition_id!=0){
        int sender_id = (partition_id - 1 + partitions) % partitions;
        if(partition_offset[partition_id]<new_partition_offset[partition_id])
          MPI_Send(p->add+partition_offset[partition_id]*p->type_size, (new_partition_offset[partition_id]-partition_offset[partition_id])*(p->type_size), MPI_CHAR, sender_id, GatherVertexArray, MPI_COMM_WORLD);
        else if(partition_offset[partition_id]>new_partition_offset[partition_id])
          MPI_Recv(p->add+new_partition_offset[partition_id]*p->type_size, (partition_offset[partition_id]-new_partition_offset[partition_id])*(p->type_size), MPI_CHAR, sender_id, GatherVertexArray,MPI_COMM_WORLD, MPI_STATUS_IGNORE);
      }else{//from the last node to the 1st node
         int sender_id = partitions-1;
         if(partition_offset[0]<new_partition_offset[0]){//send mode
           if(partition_offset[0]>=vertices || new_partition_offset[0]<=vertices)
             MPI_Send(p->add+(vertices+partition_offset[0])%vertices*p->type_size, (new_partition_offset[0]-partition_offset[0])*(p->type_size), MPI_CHAR, sender_id, GatherVertexArray, MPI_COMM_WORLD);
           else{
             MPI_Send(p->add+(vertices+partition_offset[0])%vertices*p->type_size, (0-partition_offset[0])*(p->type_size), MPI_CHAR, sender_id, GatherVertexArray, MPI_COMM_WORLD);
             MPI_Send(p->add, new_partition_offset[0]%vertices*(p->type_size), MPI_CHAR, sender_id, GatherVertexArray, MPI_COMM_WORLD);
           }
         }else if(partition_offset[0]>new_partition_offset[0]){//recv mode
           if(partition_offset[0]<=0 || new_partition_offset[0]>=0)
             MPI_Recv(p->add+(vertices+new_partition_offset[0])%vertices*p->type_size, (partition_offset[0]-new_partition_offset[0])*(p->type_size), MPI_CHAR, sender_id, GatherVertexArray, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
           else{
             MPI_Recv(p->add+(vertices+new_partition_offset[0])*p->type_size, (0-new_partition_offset[0])*(p->type_size), MPI_CHAR, sender_id, GatherVertexArray, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
             MPI_Recv(p->add, partition_offset[0]*(p->type_size), MPI_CHAR, sender_id, GatherVertexArray, MPI_COMM_WORLD,MPI_STATUS_IGNORE);
           }
         }//recv mode
      }//from the last node to the 1st node
    });
    

    right_thread.join();
    left_thread.join();
    MPI_Barrier(MPI_COMM_WORLD);

    for(int i=0;i<=partitions;i++)
      partition_offset[i]=new_partition_offset[i];
  }

  void sync_subset(VertexSubset * r){
    VertexId s1,s2,t1,t2;
    s1 = u_partition_overlap_offset[2 * partition_id];
    t1 = u_partition_overlap_offset[(2 * partition_id + 1) % (2*partitions)];
    s2 = u_partition_overlap_offset[(2 * partition_id + 2) % (2*partitions)];
    t2 = u_partition_overlap_offset[(2 * partition_id + 3) % (2*partitions)];
    unsigned long * recvdata_left, * recvdata_right;
    if(partition_id!=0) recvdata_left = new unsigned long [WORD_OFFSET(t1-s1)];
    else recvdata_left = new unsigned long [WORD_OFFSET(t1+vertices-s1)];
    if(partition_id!=partitions-1) recvdata_right = new unsigned long [WORD_OFFSET(t2-s2)];
    else recvdata_right = new unsigned long [WORD_OFFSET(t2+vertices-s2)];

      std::thread send_thread([&](){
        //send to left node, then right one
        if(partition_id!=0){
          MPI_Send((r->data)+WORD_OFFSET(s1),WORD_OFFSET(t1-s1),MPI_UNSIGNED_LONG,partition_id-1,GatherVertexArray,MPI_COMM_WORLD);
        }
        else{
          MPI_Send((r->data)+WORD_OFFSET(s1),WORD_OFFSET(vertices-s1),MPI_UNSIGNED_LONG,partitions-1,GatherVertexArray,MPI_COMM_WORLD);
          MPI_Send(r->data,WORD_OFFSET(t1),MPI_UNSIGNED_LONG,partitions-1,GatherVertexArray,MPI_COMM_WORLD);
        }
        if(partition_id!=partitions-1){
          MPI_Send((r->data)+WORD_OFFSET(s2),WORD_OFFSET(t2-s2),MPI_UNSIGNED_LONG,partition_id+1,GatherVertexArray,MPI_COMM_WORLD);
        }
        else{
          MPI_Send((r->data)+WORD_OFFSET(s2),WORD_OFFSET(vertices-s2),MPI_UNSIGNED_LONG,0,GatherVertexArray,MPI_COMM_WORLD);
          MPI_Send(r->data,WORD_OFFSET(t2),MPI_UNSIGNED_LONG,0,GatherVertexArray,MPI_COMM_WORLD);
        }
      });

      std::thread recv_thread([&](){
        //recv from right node
        //perform OR operation
        if(partition_id!=partitions-1){
          MPI_Recv(recvdata_right,WORD_OFFSET(t2-s2),MPI_UNSIGNED_LONG,partition_id+1,GatherVertexArray,MPI_COMM_WORLD,MPI_STATUS_IGNORE);
          for (size_t i=0;i<WORD_OFFSET(t2-s2);i++) ((r->data)+WORD_OFFSET(s2))[i] = ((r->data)+WORD_OFFSET(s2))[i] | recvdata_right[i];
        }
        else{
          MPI_Recv(recvdata_right,WORD_OFFSET(vertices-s2),MPI_UNSIGNED_LONG,0,GatherVertexArray,MPI_COMM_WORLD,MPI_STATUS_IGNORE);
          for (size_t i=0;i<WORD_OFFSET(vertices-s2);i++) ((r->data)+WORD_OFFSET(s2))[i] = ((r->data)+WORD_OFFSET(s2))[i] | recvdata_right[i];
          MPI_Recv(recvdata_right,WORD_OFFSET(t2),MPI_UNSIGNED_LONG,0,GatherVertexArray,MPI_COMM_WORLD,MPI_STATUS_IGNORE);
          for (size_t i=0;i<WORD_OFFSET(t2);i++) r->data[i] = r->data[i] | recvdata_right[i];
        }

        if(partition_id!=0){
          MPI_Recv(recvdata_left,WORD_OFFSET(t1-s1),MPI_UNSIGNED_LONG,partition_id-1,GatherVertexArray,MPI_COMM_WORLD,MPI_STATUS_IGNORE);
          for (size_t i=0;i<WORD_OFFSET(t1-s1);i++) ((r->data)+WORD_OFFSET(s1))[i] = ((r->data)+WORD_OFFSET(s1))[i] | recvdata_left[i];
        }
        else{
          MPI_Recv(recvdata_left,WORD_OFFSET(vertices-s1),MPI_UNSIGNED_LONG,partitions-1,GatherVertexArray,MPI_COMM_WORLD,MPI_STATUS_IGNORE);
          for (size_t i=0;i<WORD_OFFSET(vertices-s1);i++) ((r->data)+WORD_OFFSET(s1))[i] = ((r->data)+WORD_OFFSET(s1))[i] | recvdata_left[i];
          MPI_Recv(recvdata_left,WORD_OFFSET(t1),MPI_UNSIGNED_LONG,partitions-1,GatherVertexArray,MPI_COMM_WORLD,MPI_STATUS_IGNORE);
          for (size_t i=0;i<WORD_OFFSET(t1);i++) r->data[i] = r->data[i] | recvdata_left[i];
        }
      });

      send_thread.join();
      recv_thread.join();
      MPI_Barrier(MPI_COMM_WORLD);
  }
};
#endif



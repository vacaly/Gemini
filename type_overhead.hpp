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

#ifndef TYPE_HPP
#define TYPE_HPP

#include <stdint.h>
#include <algorithm>

using namespace std;

struct Empty { };

typedef int32_t VertexId;
typedef uint64_t EdgeId;

template <typename EdgeData>
struct EdgeUnit {
  VertexId src;
  VertexId dst;
  EdgeData edge_data;
} __attribute__((packed));

template <>
struct EdgeUnit <Empty> {
  VertexId src;
  union {
    VertexId dst;
    Empty edge_data;
  };
} __attribute__((packed));

template <typename EdgeData>
struct AdjUnit {
  VertexId neighbour;
  EdgeData edge_data;
} __attribute__((packed));

template <>
struct AdjUnit <Empty> {
  union {
    VertexId neighbour;
    Empty edge_data;
  };
} __attribute__((packed));

struct CompressedAdjIndexUnit {
  EdgeId index;
  VertexId vertex;
} __attribute__((packed));

template <typename EdgeData>
AdjUnit<EdgeData> * bi_find (AdjUnit<EdgeData> * b,AdjUnit<EdgeData> * e,VertexId v){
    //printf("bifind %u\n",v);
    //printf("%u,%u,%u\n",b->neighbour,e->neighbour,(e-1)->neighbour);
    int low = 0;
    int high = e-b-1;
    while(low<=high){
     int mid = low + (high-low)/2 ;
      if( (b+mid)->neighbour == v)
        return b+mid;
      else if ((b+mid)->neighbour > v)
        high = mid-1;
      else
        low = mid+1;
    }
    return b+low;
}

template <typename EdgeData>
bool cmpfunc (AdjUnit<EdgeData> i,AdjUnit<EdgeData> j) {
    return (i.neighbour < j.neighbour);
}

template <typename EdgeData>
struct VertexAdjList {
    AdjUnit<EdgeData> * begin;
    AdjUnit<EdgeData> * end;
    VertexAdjList() : begin(nullptr), end(nullptr) { }
    VertexAdjList(AdjUnit<EdgeData> * stationary_begin,
                  AdjUnit<EdgeData> * stationary_end,
                  VertexId floating_offset_beg,
                  VertexId floating_offset_end,
                  VertexId vertices)
    : begin(stationary_begin), end(stationary_end){
        if(floating_offset_beg<0 || floating_offset_end>vertices){
          floating_offset_beg = floating_offset_beg<0 ? floating_offset_beg+vertices : floating_offset_beg;
          floating_offset_end = floating_offset_end>vertices ? floating_offset_end-vertices : floating_offset_end;
        }
        if(stationary_begin->neighbour >= floating_offset_beg){
            //printf("end->nei:%u,%u\n",stationary_end->neighbour,(stationary_end-1)->neighbour);

            if((stationary_end-1)->neighbour < floating_offset_end){
                //-------F---S****S--F-------
                begin = stationary_begin;
                end   = stationary_end;
            }else if(stationary_begin->neighbour >= floating_offset_end){
                //-------F-------F--S---S---
                //printf("case1\n");
                begin = nullptr;
                end   = nullptr;
            }else{
                //-------F---S******F---S----
                begin = stationary_begin;
                end   = bi_find(stationary_begin,stationary_end,floating_offset_end);
            }
        }else{
            //printf("end->nei:%u,\n",stationary_end->neighbour,(stationary_end-1)->neighbour);

            if((stationary_end-1)->neighbour < floating_offset_beg){
                //---S---S-F--------F-------
                //printf("case2\n");
                begin = nullptr;
                end   = nullptr;
            }else if((stationary_end-1)->neighbour < floating_offset_end){
                //---S----F*****S----F-------
                begin = bi_find(stationary_begin,stationary_end,floating_offset_beg);
                end   = stationary_end;
            }else{
                //---S----F********F---S-----
                begin = bi_find(stationary_begin,stationary_end,floating_offset_beg);
                end   = bi_find(stationary_begin,stationary_end,floating_offset_end);
            }
        }
        
    }
    VertexAdjList(AdjUnit<EdgeData> * begin,
                  AdjUnit<EdgeData> * end
                ): begin(begin), end(end) { }
    /*
    VertexAdjList(AdjUnit<EdgeData> * stationary_begin,
                  AdjUnit<EdgeData> * stationary_end){
        sort(stationary_begin,stationary_end,cmpfunc<EdgeData>);
    }*/
};

struct VertexArray {
  char * add;
  unsigned type_size;
  struct VertexArray * next;
  VertexArray(char * addT,unsigned sizeofT) {
    add = addT;
    type_size = sizeofT;
    next = NULL;
  }
};

struct VertexHead {
  VertexArray * headaddr;
  VertexHead():headaddr(NULL) {};
};
#endif

#include <bits/stdc++.h>
 
using namespace std;

class TensorInternal {
    private:
        vector<float> v;
        vector<int> shape;
    public:
        TensorInternal(vector<int>& shape_in, float fill=0.0f){
            size_t num_eles = 1;
            for(auto x : shape_in){
                shape.push_back(x);
                num_eles *= x;
            }

            v.assign(num_eles, fill);
        }
        // operator=(float fill){
        //     for(auto &x : v)
        //         x = fill;
        //     return (*this);
        // }
        operator[](int idx){
            return v[idx];
        }

        
};
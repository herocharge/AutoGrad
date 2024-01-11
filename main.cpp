#include <bits/stdc++.h>

using namespace std;

class AD;
struct Node{
    int idx;
    bool is_leaf = true;
    vector<float>& v;
    vector<float>& dv;
    Node(vector<float>& v, vector<float>& dv) : v(v), dv(dv){}
};

struct Edge
{
    int idx;
    vector<float>& dd;
    Edge(int idx, vector<float>& dd) : idx(idx), dd(dd) {}
};


template<typename T>
class Tensor : public vector<T>{
    public:
        bool grad_required = false;
        vector<T> grad;
        bool is_leaf = true;
        AD* comp_graph; // yuck
        Node* cg_node;

        Tensor<T>(T init, AD* cg) : vector<T>(init){
            comp_graph = cg;
        }
        Tensor<T>(AD* cg) : vector<T>(){
            comp_graph = cg;
        }

        void need_grad(){
            grad_required = true;
            
            cg_node = comp_graph->add_leaf(*this);

            size_t size = this->size();
            cout<<"[HERO]"<<size<<endl;
            grad.assign(size, 0);
        }

        Tensor<T> operator*(int op1){
            auto res_ptr = new Tensor<T>(comp_graph);
            Tensor<T>& result = (*res_ptr);
            result.grad_required |= this->grad_required;
            auto dd = new vector<float>();
            for(auto x : *this){
                result.push_back(x * op1);
                if(result.grad_required)
                    (*dd).push_back(op1);
            }
            if(result.grad_required){
                // result.need_grad();
                size_t size = result.size();
                result.grad.assign(size, 0);
                result.cg_node =  comp_graph->add_intermediate(result, cg_node, dd);
            }
            return *res_ptr;
        }
        Tensor<T> operator+(int op1){
            auto res_ptr = new Tensor<T>(comp_graph);
            Tensor<T>& result = (*res_ptr);
            result.grad_required |= this->grad_required;
            auto dd = new vector<float>();
            for(auto x : *this){
                result.push_back(x + op1);
                if(result.grad_required)
                    (*dd).push_back(1);
            }
            if(result.grad_required){
                // result.need_grad();
                size_t size = result.size();
                result.grad.assign(size, 0);
                result.cg_node =  comp_graph->add_intermediate(result, cg_node, dd);
            }
            return *res_ptr;
        }
        Tensor<T> operator-(int op1){
            auto res_ptr = new Tensor<T>(comp_graph);
            Tensor<T>& result = (*res_ptr);
            result.grad_required |= this->grad_required;
            auto dd = new vector<float>();
            for(auto x : *this){
                result.push_back(x - op1);
                if(result.grad_required)
                    (*dd).push_back(1);
            }
            if(result.grad_required){
                // result.need_grad();
                size_t size = result.size();
                result.grad.assign(size, 0);
                result.cg_node = comp_graph->add_intermediate(result, cg_node, dd);
            }
            return *res_ptr;
        }
        Tensor<T> operator/(int op1){
            auto res_ptr = new Tensor<T>(comp_graph);
            Tensor<T>& result = (*res_ptr);
            result.grad_required |= this->grad_required;
            auto dd = new vector<float>();
            for(auto x : *this){
                result.push_back(x / op1);
                if(result.grad_required)
                    (*dd).push_back(1.0/op1);
            }
            if(result.grad_required){
                // result.need_grad();
                size_t size = result.size();
                result.grad.assign(size, 0);
                result.cg_node =  comp_graph->add_intermediate(result, cg_node, dd);
            }
            return *res_ptr;
        }

        Tensor<T> operator*(float op1){
            auto res_ptr = new Tensor<T>(comp_graph);
            Tensor<T>& result = (*res_ptr);
            result.grad_required |= this->grad_required;
            auto dd = new vector<float>();
            for(auto x : *this){
                result.push_back(x * op1);
                if(result.grad_required)
                    (*dd).push_back(op1);
            }
            if(result.grad_required){
                // result.need_grad();
                size_t size = result.size();
                result.grad.assign(size, 0);
                result.cg_node =  comp_graph->add_intermediate(result, cg_node, dd);
            }
            return *res_ptr;
        }
        Tensor<T> operator+(float op1){
            auto res_ptr = new Tensor<T>(comp_graph);
            Tensor<T>& result = (*res_ptr);
            result.grad_required |= this->grad_required;
            auto dd = new vector<float>();
            for(auto x : *this){
                result.push_back(x + op1);
                if(result.grad_required)
                    (*dd).push_back(1);
            }
            if(result.grad_required){
                // result.need_grad();
                size_t size = result.size();
                result.grad.assign(size, 0);
                result.cg_node =  comp_graph->add_intermediate(result, cg_node, dd);
            }
            return *res_ptr;
        }
        Tensor<T> operator-(float op1){
            auto res_ptr = new Tensor<T>(comp_graph);
            Tensor<T>& result = (*res_ptr);
            result.grad_required |= this->grad_required;
            auto dd = new vector<float>();
            for(auto x : *this){
                result.push_back(x - op1);
                if(result.grad_required)
                    (*dd).push_back(1);
            }
            if(result.grad_required){
                // result.need_grad();
                size_t size = result.size();
                result.grad.assign(size, 0);
                result.cg_node = comp_graph->add_intermediate(result, cg_node, dd);
            }
            return *res_ptr;
        }
        Tensor<T> operator/(float op1){
            auto res_ptr = new Tensor<T>(comp_graph);
            Tensor<T>& result = (*res_ptr);
            result.grad_required |= this->grad_required;
            auto dd = new vector<float>();
            for(auto x : *this){
                result.push_back(x / op1);
                if(result.grad_required)
                    (*dd).push_back(1.0/op1);
            }
            if(result.grad_required){
                // result.need_grad();
                size_t size = result.size();
                result.grad.assign(size, 0);
                result.cg_node =  comp_graph->add_intermediate(result, cg_node, dd);
            }
            return *res_ptr;
        }
        
        Tensor<T> operator+(const Tensor<T>& op1){
            assert(op1.size() == (*this).size());
            size_t len = op1.size();
            if(&op1 == this){
                return (*this) * 2;
            }
            auto res_ptr = new Tensor<T>(comp_graph);
            Tensor<T>& result = (*res_ptr);
            result.grad_required |= this->grad_required;

            auto dd1 = new vector<float>();
            auto dd2 = new vector<float>();
            for(size_t i = 0; i < len; i++){
                result.push_back((*this)[i] + op1[i]);
                if(result.grad_required){
                    (*dd1).push_back(1);
                    (*dd2).push_back(1);
                }
            }
            if(result.grad_required){
                // result.need_grad();
                size_t size = result.size();
                result.grad.assign(size, 0);
                if(op1.grad_required)
                    result.cg_node =  comp_graph->add_intermediate(result, cg_node, dd1, op1.cg_node, dd2);
                else
                    result.cg_node =  comp_graph->add_intermediate(result, cg_node, dd1);
            }
            return *res_ptr;
        }
        Tensor<T> operator-(Tensor<T>& op1){
            assert(op1.size() == (*this).size());
            size_t len = op1.size();
            auto res_ptr = new Tensor<T>(comp_graph);
            Tensor<T>& result = (*res_ptr);
            result.grad_required |= this->grad_required;

            auto dd1 = new vector<float>();
            auto dd2 = new vector<float>();
            for(size_t i = 0; i < len; i++){
                result.push_back((*this)[i] - op1[i]);
                if(result.grad_required){
                    (*dd1).push_back(1);
                    (*dd2).push_back(-1);
                }
            }
            if(result.grad_required){
                // result.need_grad();
                size_t size = result.size();
                result.grad.assign(size, 0);
                if(op1.grad_required)
                    result.cg_node =  comp_graph->add_intermediate(result, cg_node, dd1, op1.cg_node, dd2);
                else
                    result.cg_node =  comp_graph->add_intermediate(result, cg_node, dd1);
            }
            return *res_ptr;
        }
        Tensor<T> operator*(Tensor<T>& op1){
            assert(op1.size() == (*this).size());
            if(&op1 == this){
                return (*this).power(2);
            }
            size_t len = op1.size();
            auto res_ptr = new Tensor<T>(comp_graph);
            Tensor<T>& result = (*res_ptr);
            result.grad_required |= this->grad_required;

            auto dd1 = new vector<float>();
            auto dd2 = new vector<float>();
            for(size_t i = 0; i < len; i++){
                result.push_back((*this)[i] * op1[i]);
                if(result.grad_required){
                    (*dd1).push_back(op1[i]);
                    (*dd2).push_back((*this)[i]);
                }
            }
            if(result.grad_required){
                // result.need_grad();
                size_t size = result.size();
                result.grad.assign(size, 0);
                if(op1.grad_required)
                    result.cg_node =  comp_graph->add_intermediate(result, cg_node, dd1, op1.cg_node, dd2);
                else
                    result.cg_node =  comp_graph->add_intermediate(result, cg_node, dd1);
            }
            return *res_ptr;
        }
        Tensor<T> operator/(Tensor<T>& op1){
            assert(op1.size() == (*this).size());
            size_t len = op1.size();
            Tensor<T> result = *this;
            for(size_t i = 0; i < len; i++){
                result[i] = result[i] / op1[i];
            }
            return result;
        }
        Tensor<T> power(int op1){
            auto res_ptr = new Tensor<T>(comp_graph);
            Tensor<T>& result = (*res_ptr);
            result.grad_required |= this->grad_required;
            auto dd = new vector<float>();
            for(auto x : *this){
                result.push_back(std::pow(x, op1));
                if(result.grad_required)
                    (*dd).push_back(op1 * (std::pow(x, op1-1)));
            }
            if(result.grad_required){
                // result.need_grad();
                size_t size = result.size();
                result.grad.assign(size, 0);
                result.cg_node =  comp_graph->add_intermediate(result, cg_node, dd);
            }
            return *res_ptr;
        }
};



class AD{
    private:
        vector<Node *> nodes;
        vector<vector<Edge>> adj;

        vector<float> vbar;
    public:

    Node* add_leaf(Tensor<float>& leaf){
        Node* node = new Node(leaf, leaf.grad);
        node->idx = adj.size();
        node->is_leaf = true;
        nodes.push_back(node);
        adj.push_back(vector<Edge>());
        return node;
    }

    // Node* add_leaf(Tensor<int>& leaf){
    //     Node* node = new Node(leaf, leaf.grad);
    //     node->leaf_idx = adj.size();
    //     node->is_leaf = true;
    //     leaves.push_back(node);
    //     adj.push_back(vector<Edge>());
    //     return node;
    // }

    Node* add_intermediate(Tensor<float>& interm, Node* dep1, vector<float>* dd1, Node* dep2 = nullptr, vector<float>* dd2 = nullptr){
        cout<<"[PTR]"<<&interm<<endl;
        Node* node = new Node(interm, interm.grad);
        node->idx = adj.size();
        node->is_leaf = false;
        nodes.push_back(node);
        adj.push_back(vector<Edge>());

        int idx1 = dep1->idx;
        adj[node->idx].push_back(Edge(idx1, *dd1));
        cout<<dep2<<endl;
        if(dep2 != nullptr){
            int idx2 = dep2->idx;
            adj[node->idx].push_back(Edge(idx2, *dd2));
        }

        return node;
    }

    void backward(){

        for(int i = 0; i < adj.size(); i++){
            cout<<i<<": "<<nodes[i]->is_leaf<<": ";
            for(auto edge : adj[i]){
                cout<<edge.idx<<" ";
            }
            cout<<endl;
        }
        int curr = adj.size() - 1;
        // TODO: search for nodes with zero incoming instead of taking last node


        int len = nodes[curr]->v.size();
        cout<<"LEN "<<curr<<endl;
        // cout<<"pts "<<(&(nodes[0]->v))<<" "<<(&(nodes[1]->v))<<" "<<(&(nodes[2]->v))<<" "<<(&(nodes[3]->v))<<endl;
        // cout<<"ptr "<<((nodes[0]))<<" "<<((nodes[1]))<<" "<<((nodes[2]))<<" "<<((nodes[3]))<<endl;
        for(int i = 0; i < len; i++){
            nodes[curr]->dv[i] = 1;
        }
        queue<int> q;
        q.push(curr);
        while(!q.empty()){
            int curr = q.front();
            cout<<"CURR="<<curr<<endl;
            q.pop();
            for(auto edge : adj[curr]){
                // TODO: implement for vector/matrix correctness
                int len = edge.dd.size();
                int idx = edge.idx;
                cout<<edge.dd.size()<<" "<<nodes[idx]->dv.size()<<nodes[curr]->dv.size()<<endl;
                for(int i = 0; i < len; i++){
                    cout<<"vals["<<idx<<"] "<<nodes[idx]->dv[i]<<"+="<<nodes[curr]->dv[i]<<"*"<<edge.dd[i]<<endl;

                    nodes[idx]->dv[i] += nodes[curr]->dv[i] * edge.dd[i];
                }
                q.push(edge.idx);
            }
        }
    }

    void zero_grad(){
        for(Node* node : nodes){
            // if(node == NULL)continue;
            // cout<<
            for(auto &x : node->dv){
                x = 0;
            }
        }
    }



    template <typename T>
    Tensor<T> tensor(T init){
        return Tensor<T>(init, this);
    }

    template <typename T>
    Tensor<T> tensor(){
        return Tensor<T>(this);
    }
};

template <typename T>
Tensor<T> operator-(int left, Tensor<T>& right){
    return (right - left) * -1;
}
template <typename T>
Tensor<T> operator+(int left, Tensor<T>& right){
    return right + left;
}
template <typename T>
Tensor<T> operator*(int left, Tensor<T>& right){
    return right * left;
}
template <typename T>
Tensor<T> operator-(float left, Tensor<T>& right){
    return (right - left) * -1;
}
template <typename T>
Tensor<T> operator+(float left, Tensor<T>& right){
    return right + left;
}
template <typename T>
Tensor<T> operator*(float left, Tensor<T>& right){
    return right * left;
}


// gptgen
std::vector<std::vector<std::string>> readCSV(const std::string& filename) {
    std::vector<std::vector<std::string>> data;

    // Open the CSV file
    std::ifstream file(filename);
    
    if (!file.is_open()) {
        std::cerr << "Error opening file: " << filename << std::endl;
        return data;  // Return an empty vector if the file couldn't be opened
    }

    std::string line;
    
    // Read each line from the file
    while (std::getline(file, line)) {
        std::vector<std::string> row;
        std::stringstream ss(line);
        std::string cell;

        // Split the line into cells using a comma as a delimiter
        while (std::getline(ss, cell, ',')) {
            row.push_back(cell);
        }

        data.push_back(row);
    }

    file.close();

    return data;
}

void iris(){
    AD ad;

    vector<int> layer_dims = {4, 3};
    vector<vector<vector<Tensor<float>>>> weights;

    // array of inp dim x out dim matrix
    for(int k = 1; k < layer_dims.size(); k++){  
        vector<vector<Tensor<float>>> tmp2;  
        for(int i = 0; i < layer_dims[k-1]; i++){
            vector<Tensor<float>> tmp;
            for(int j = 0; j < layer_dims[k]; j++){
                tmp.push_back(ad.tensor<float>());
                tmp[j].push_back(0);
                tmp[j].need_grad();
            }
            tmp2.push_back(tmp);
        }
        weights.push_back(tmp2);
    }


    // Get data
    auto data = readCSV("iris.csv");

    for(auto row : data){
        for(auto col : row){
            cout<<col<<"\t";
        }
        cout<<endl;
    }

    // split into x, y
    vector<vector<float>> data_x;
    vector<int> data_y;
    bool first = true;
    for(auto row : data){
        if(first){
            first = false;
            continue;
        }
        data_x.push_back({
            stof(row[0]),
            stof(row[1]),
            stof(row[2]),
            stof(row[3]),
        });
        if(row[0] == "Setosa"){
            data_y.push_back(0);
        }
        else if(row[0] == "Versicolor"){
            data_y.push_back(1);
        }
        else{
            data_y.push_back(2);
        }
    }

    // TODO: split into train and test

    
    int epochs = 10;
    while (epochs--)
    {
        cout<<epochs<<endl;
        for(auto X : data_x){
            vector<vector<Tensor<float>>> activations;
            for(int i = 0; i < layer_dims.size(); i++){
                vector<Tensor<float>> tmp;
                for(int j = 0; j < layer_dims[i]; j++){
                    tmp.push_back(ad.tensor<float>());
                    if(i == 0){
                        // cout<<X[j]<<endl;
                        tmp[j].push_back(X[j]);
                    }
                    else{
                        tmp[j].push_back(0);
                    }
                }
                activations.push_back(tmp);
            }
            for(int k = 1; k < layer_dims.size(); k++){
                for(int j = 0; j < layer_dims[k]; j++){
                    for(int i = 0; i < layer_dims[k-1]; i++){
                        activations[k][j] = activations[k][j] +  weights[k-1][i][j] * activations[k-1][i];
                    }
                }
            }
            // ad.backward();
            ad.zero_grad();

        }
    }
       

    

}

int32_t main(){
    AD ad;
    auto a = ad.tensor<float>();
    auto c = ad.tensor<float>();
    c.push_back(5);
    a.push_back(10);
    a.push_back(10);
    a.need_grad();

    // a.push_back(10);
    for(int i = 0; i < 10; i++  )
    a = 2 * a;
    // auto c = b + b;
    // auto d = a;
    // auto c = (b / 6);
    ad.backward();
    for(auto x : a)cout<<x<<" ";cout<<endl;
    for(auto x : a.grad)cout<<x<<" ";cout<<endl;
    
    iris();
}


// TODO: need to delete the extra pointers


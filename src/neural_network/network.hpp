#ifndef COGNOSCO_NETWORK_HPP
#define COGNOSCO_NETWORK_HPP
namespace cognosco{
  class Network{
    public:
    Network();
    Network(int layer_count, ...);
    Network(std::vector<int> layers);
    Network(const Network& copy_net);
    ~Network();
    private:
    int n_layers;
    std::vector<matrix<double>> weight_matrix, bias_matrix;
  };
}
#endif

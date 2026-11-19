#ifndef N_H
#define N_H

typedef struct DataSet {
    double*** data;
    int E;
    int length;
} DataSet;

class N {
public:
    double* I;
    double** H;
    double* O;
    double*** W;
    double lr;
    int layers;

    int inSize_, hidSize_, outSize_;

    N(int i, int h, int layers_, int o);
    ~N();

    double s(double x);
    double sd(double x);
    double tanh(double x);
    double tanh_d(double x);

    double* softmax(double* z);
    double neuron(double* X, double* Wv);
    double* forward(double* X);
    double backprop(double* T);

    void train(DataSet D);
    void train(DataSet D, int E);

    char* save();
    void load(const char* data);

private:
    static unsigned long rngState;

    static double rand_next();
    static double exp_(double x);
    static double tanh_(double x);

    static int len(double* D);
    static int len(double** D);

    static int intToStr(char* buf, int v);
    static int doubleToStr(char* buf, double v);
};

#endif
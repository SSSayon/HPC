#ifndef SpMM_OPT_H
#define SpMM_OPT_H
#include "spmm_base.h"

/*
    Variables we can access:
    
    int *d_ptr = NULL;
    int *d_idx = NULL;
    float *d_val = NULL;

    int feat_in = 0;

    int num_v = 0;
    int num_e = 0;

    dim3 grid;
    dim3 block;
*/

struct TRUNCATED_LINE {
    int row;
    int begin, end;
};

#define max_element_per_line 30 // 25 10
#define combined_size 2 //         2  3
struct COMBINED_LINE {
    int rows[combined_size];
    int begins[combined_size];
    int ends[combined_size];
};

struct RowInfo {
    int index;
    int begin;
    int end;
};

class SpMMOpt : public SpMM
{
public:
    SpMMOpt(int *dev_out_ptr, int *dev_out_idx, int out_num_v, int out_num_e, int out_feat_in) : SpMM(dev_out_ptr, dev_out_idx, out_num_v, out_num_e, out_feat_in) {}
    SpMMOpt(CSR *g, int out_feat_in) : SpMM(g, out_feat_in) {}
    ~SpMMOpt() {
        if (d_truncated_lines) checkCudaErrors(cudaFree(d_truncated_lines));
    }

    virtual void preprocess(float *vin, float *vout);
    virtual void run(float *vin, float *vout);

private:
    int num_truncated_lines = 0;
    TRUNCATED_LINE *d_truncated_lines;

    dim3 grid2;
    dim3 block2;
    int num_combined_lines = 0;
    COMBINED_LINE *d_combined_lines;

    void preprocess_truncated_line(int truncated_step);
    void preprocess_trunc_combined_line(int truncated_step);
};

#endif
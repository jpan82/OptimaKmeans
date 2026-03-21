// Data Loader header file for k-means algorithm 
#ifndef DATALOADER_H
#define DATALOADER_H

// Load data from a binary file or a CSV file
// Returns 0 on success, -1 on failure
int load_data_bin(const char* filename, float** data, int* n, int* d);
int load_data_csv(const char* filename, float** data, int* n, int* d);

// Free the allocated data
void free_data(float* data);

#endif // DATALOADER_H
#include <stdio.h>
#include <cuda.h>

#define THREADS 256   // Threads per block

struct device_Struct {
    int device;
    int deviceCount;
    unsigned int number_values;
    unsigned int number_values_interior;
    float *data_interior_in;
    float *data_boundary_left_in;
    float *data_boundary_right_in;
    float *data_halo_left_in;
    float *data_halo_right_in;
    float *data_interior_out;
    float *data_boundary_left_out;
    float *data_boundary_right_out;
    float *data_halo_left_out;
    float *data_halo_right_out;
  };

// Jacobi stencil kernel: each interior point is updated as the average of its six neighbors.
__global__ void jacobi3D(struct device_Struct this_device, int NX, int NY, int NZ, int device, int deviceCount) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    //guard
    if(i >=  NX * NY * NZ) return;
    //compute indexing for readabilty
    int z = i / (NX * NY);
    int r = (i % (NX * NY));
    int x = r % NX;
    int y = r / NY;
    
    //for devices with a left neighbor values of z=0 are in boundary_left_in and for devices with right neigbor vaues of z=NZ-1 are in boundary_right_in
    float *input_addr;
    float *output_addr;
    if(z == 0 && device != 0){
        input_addr = this_device.data_boundary_left_in;
        output_addr = this_device.data_boundary_left_out;
    } else if(z == NZ-1 && device != deviceCount - 1){
        input_addr = this_device.data_boundary_right_in;
        output_addr = this_device.data_boundary_right_out;
    }else{
        input_addr = this_device.data_interior_in;
        output_addr = this_device.data_interior_out;
    }
    output_addr[i] = 0;
    int count = 0;
    //TODO: compute boundaries in seperate kernel to get around if-else-statements
    // got north neighbor
    if(y > 0){
        output_addr[i] += input_addr[i - NX];
        count ++;
    }
    // got south neighbor
    if(y < NY - 1){
        output_addr[i] += input_addr[i+NX];
        count ++;
    }
    // got west neighbor
    if(x > 0){
        output_addr[i] += input_addr[i-1];
        count ++;
    }
    // got east neighbor
    if(x < NX - 1){
        output_addr[i] += input_addr[i+1];
        count ++;
    }
    //got front neighbor
    if(device == 0 && z == 0){
        //no front neighbor
    }else if(device == 0 && z > 0){
        //front neighbor as usual in interior
        output_addr[i] += input_addr[i - NX * NY];
        count ++;
    }else{
        //front neighbor is in halo
        output_addr[i] += this_device.data_halo_left_in[y * NY + x];
        count ++;
    }
    //got back neighbor
    if(device == device-1 && z == NZ-1){
        //no back neighbor
    }else if(device == device-1 && z < NZ-1){
        //back neighbor as usual in interior
        output_addr[i] += input_addr[i + NX * NY];
        count ++;
    }else{
        //back neighbor is in halo
        output_addr[i] += this_device.data_halo_right_in[y * NY + x];
        count ++;
    }
}

// Multi-GPU Jacobi3D
// halo exchange via host and seperate in/out buffers
void jacobi3D(float* h_A, int NX, int NY, int NZ, int TSTEPS){
    unsigned int N = NX * NY * NZ;
    int deviceCount;
    cudaGetDeviceCount(&deviceCount);
    struct device_Struct devices[deviceCount];
    
    //slice matrix and copy data to devices
    unsigned int slice_width = NZ / deviceCount;
    for(int device = 0; device<deviceCount; device++){
        cudaSetDevice(device);
        printf("Device %d is being set up.\n", device);
        unsigned int number_values = device * slice_width * NX * NY;
        if(device == deviceCount-1){
            number_values += (NZ % deviceCount) * NX * NY;
        }
        devices[device].number_values = number_values;

        int number_values_interior = number_values - 2 * NX * NY;
        //edge slices have one boundary less left block
        if(device == 0 || device == deviceCount-1){
            number_values_interior += NX * NY;
        }
        devices[device].number_values_interior = number_values_interior;

        cudaMalloc(&devices[device].data_interior_in, number_values_interior * sizeof(float));
        cudaMalloc(&devices[device].data_interior_out, number_values_interior * sizeof(float));
        cudaMemcpy(&devices[device].data_interior_in, h_A + (device != 0) * NX * NY + device * slice_width * NX * NY, number_values_interior * sizeof(float), cudaMemcpyHostToDevice);
        //has left neighbor
        if(device != 0){
            cudaMalloc(&devices[device].data_boundary_left_in, NX * NY * sizeof(float));
            cudaMalloc(&devices[device].data_halo_left_in, NX * NY * sizeof(float));
            cudaMalloc(&devices[device].data_boundary_left_out, NX * NY * sizeof(float));
            cudaMalloc(&devices[device].data_halo_left_out, NX * NY * sizeof(float));
            cudaMemcpy(devices[device].data_boundary_left_in, h_A + device * slice_width * NX * NY, NX * NY * sizeof(float), cudaMemcpyHostToDevice);
            cudaMemcpy(devices[device].data_halo_left_in, h_A + (device * slice_width - 1) * NX * NY, NX * NY * sizeof(float), cudaMemcpyHostToDevice);
        }
        //has right neighbor
        if(device != deviceCount -1){
            cudaMalloc(&devices[device].data_boundary_right_in, NX * NY * sizeof(float));
            cudaMalloc(&devices[device].data_halo_right_in, NX * NY * sizeof(float));
            cudaMalloc(&devices[device].data_boundary_right_out, NX * NY * sizeof(float));
            cudaMalloc(&devices[device].data_halo_right_out, NX * NY * sizeof(float));
            cudaMemcpy(devices[device].data_boundary_right_in, h_A + ((device + 1) * slice_width - 1) * NX * NY, NX * NY * sizeof(float), cudaMemcpyHostToDevice);
            cudaMemcpy(devices[device].data_halo_right_in, h_A + (device + 1) * slice_width * NX * NY, NX * NY * sizeof(float), cudaMemcpyHostToDevice);
        }
    }

    float* h_buffer = (float *)malloc(NX * NY * sizeof(float)); //for copying data from device to host to device
    for(int i = 0; i<TSTEPS; i++){
        //let GPUs compute
        for(int device = 0; device < deviceCount; device++){
            cudaSetDevice(device);
            //COMPUTE
            int blocks = (N + THREADS - 1) / THREADS;
            unsigned int number_values = device * slice_width * NX * NY;
            if(device == deviceCount-1){
                number_values += (NZ % deviceCount) * NX * NY;
            }
            jacobi3D<<<blocks, THREADS>>>(devices[device], NX, NY, slice_width + (device == deviceCount-1) * (NZ % deviceCount), device, deviceCount);
        }

        //wait for gpus to finish
        for(int device = 0; device<deviceCount; device++){
            cudaSetDevice(device);
            cudaDeviceSynchronize();
        }

        //exchange halos
        //TODO: Pointer Swapping
        for(int device = 0; device < deviceCount; device++){
            cudaSetDevice(device);
            //has left neighbor
            if(device != 0){
                cudaSetDevice(device-1);
                cudaMemcpy(h_buffer, devices[device-1].data_boundary_right_out, NX * NY * sizeof(float), cudaMemcpyDeviceToHost);
                cudaSetDevice(device);
                cudaMemcpy(devices[device].data_halo_left_in, h_buffer, NX * NY * sizeof(float), cudaMemcpyHostToDevice);
            }
            //has right neighbor
            if(device != device - 1){
                cudaSetDevice(device+1);
                cudaMemcpy(h_buffer, devices[device+1].data_boundary_left_out, NX * NY * sizeof(float), cudaMemcpyDeviceToHost);
                cudaSetDevice(device);
                cudaMemcpy(devices[device].data_halo_right_in, h_buffer, NX * NY * sizeof(float), cudaMemcpyHostToDevice);
            }
            //Pointer swapping here
        }
    }

    //copy data back to host memory
    float *write_addr = h_A;
    for(int device = 0; device < deviceCount; device++){
        cudaSetDevice(device);
        //copy back left boundary
        if(device != 0){
            cudaMemcpy(write_addr, devices[device].data_boundary_left_out, NX * NY * sizeof(float), cudaMemcpyDeviceToHost);
            write_addr += NX * NY;
        }
        //copy back interior
        cudaMemcpy(write_addr, devices[device].data_interior_out, devices[device].number_values_interior * sizeof(float), cudaMemcpyDeviceToHost);
        write_addr += devices[device].number_values_interior;
        //copy back right boundary
        if(device != deviceCount-1){
            cudaMemcpy(write_addr, devices[device].data_boundary_right_out, NX * NY * sizeof(float), cudaMemcpyDeviceToHost);
            write_addr += NX * NY;
        }
        //free memory
        cudaFree(devices[device].data_interior_in);
        cudaFree(devices[device].data_boundary_left_in);
        cudaFree(devices[device].data_boundary_right_in);
        cudaFree(devices[device].data_halo_left_in);
        cudaFree(devices[device].data_halo_right_in);
        cudaFree(devices[device].data_interior_in);
        cudaFree(devices[device].data_boundary_left_out);
        cudaFree(devices[device].data_boundary_right_out);
        cudaFree(devices[device].data_halo_left_out);
        cudaFree(devices[device].data_halo_right_out);
        printf("Device %d successfully copied result back to host.\n", device);
    }

}

int main() {
    int NX = 128;
    int NY = 128;
    int NZ = 128;
    int TSTEPS = 100;
    unsigned int N = NX * NY * NZ;
    float *h_A = (float*)malloc(N * sizeof(float));

    // Initialize A. Layout A: [x=0, y=0, z=0], [x=1, y=0, z=0], ..., [x=NX-1, y=0, z=0], [x=0, y=1, z=0], ..., [x=NX-1, y=NY-1, z=0], [x=0, y=0, z=1], [x=1, y=0, z=1], ..., [x=NX-1, y=NY-1, z=NZ-1]
    for (int i = 0; i < N; i++) {
        h_A[i] = i;
    }

    jacobi3D(h_A, NX, NY, NZ, TSTEPS);

    for(int i = 0; i < N; i += N/10){
        printf("%f ", h_A[i]);
    }

    free(h_A);

    return 0;
}

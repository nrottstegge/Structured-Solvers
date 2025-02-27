gj#include <stdio.h>
#include <cuda.h>

#define THREADS 256   // Threads per block

struct device_Struct {
    int device;
    int deviceCount;
    unsigned int number_values;
    unsigned int number_values_interior;
    float *data_in;
    float *data_out;
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
__global__ void jacobi3D_kernel(struct device_Struct this_device, int NX, int NY, int NZ, int device, int deviceCount) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    //guard
    if(i >=  NX * NY * NZ) return;
    //compute indexing for readabilty
    int z = i / (NX * NY);
    int r = (i % (NX * NY));
    int x = r % NX;
    int y = r / NX;
    
    //for devices with a left neighbor values of z=0 are in boundary_left_in and for devices with right neigbor vaues of z=NZ-1 are in boundary_right_in
    float *input_addr = this_device.data_in;
    float *output_addr = this_device.data_out;
    int count = 0;
    float result = 0;
    //TODO: compute boundaries in seperate kernel to get around if-else-statements
    // got north neighbor
    if(y > 0){
        result += input_addr[i - NX];
        count ++;
    }
    // got south neighbor
    if(y < NY - 1){
        result += input_addr[i + NX];
        count ++;
    }
    // got west neighbor
    if(x > 0){
        result += input_addr[i-1];
        count ++;
    }
    // got east neighbor
    if(x < NX - 1){
        result += input_addr[i+1];
        count ++;
    }
    //got front neighbor
    if(z == 0){ //front neighbor might be in halo
        if(device > 0){ //device 0 has no front neighbor in z==0
            //front neighbor is in halo
            result += this_device.data_halo_left_in[y * NX + x];
            count++;
        }
    }else{
        //front neighbor as usual in interior
        result += input_addr[i - NX * NY];
        count ++;
    }
    //got back neighbor
    if(z == NZ-1){ //back neighbor might be in halo
        if(device < deviceCount-1){ //device deviceCount-1 has no back neighbor in z==NZ-1
            //back neighbor is in halo
            result += this_device.data_halo_right_in[y * NX + x];
            count++;
        }
    }else{
        //back neighbor as usual in interior
        result += input_addr[i + NX * NY];
        count ++;
    }
    result /= count;
    output_addr[i] = result;
}

// Multi-GPU Jacobi3D
// halo exchange via host and seperate in/out buffers
void jacobi3D(float* h_A, int NX, int NY, int NZ, int TSTEPS){
    int deviceCount;
    cudaGetDeviceCount(&deviceCount);
    struct device_Struct devices[deviceCount];
    
    //slice matrix and copy data to devices
    unsigned int slice_width = NZ / deviceCount;
    for(int device = 0; device<deviceCount; device++){
        cudaSetDevice(device);
        printf("Device %d is being set up.\n", device);
        unsigned int number_values = slice_width * NX * NY;
        if(device == deviceCount-1){
            number_values += (NZ % deviceCount) * NX * NY;
        }
        devices[device].number_values = number_values;

        unsigned int number_values_interior = number_values - 2 * NX * NY;
        //edge slices have one boundary less left block
        if(device == 0 || device == deviceCount-1){
            number_values_interior += NX * NY;
        }
        devices[device].number_values_interior = number_values_interior;

        //copy whole slice to device and define boundary and interior pointers
        cudaMalloc(&devices[device].data_in, devices[device].number_values * sizeof(float));
        cudaMalloc(&devices[device].data_out, devices[device].number_values * sizeof(float));
        devices[device].data_boundary_left_in = devices[device].data_in;
        devices[device].data_interior_in = devices[device].data_in + (device != 0) * NX * NY;
        devices[device].data_boundary_right_in = devices[device].data_in + (slice_width - 1) * NX * NY;
        devices[device].data_boundary_left_out = devices[device].data_out;
        devices[device].data_interior_out = devices[device].data_out + (device != 0) * NX * NY;
        devices[device].data_boundary_right_out = devices[device].data_out + (slice_width - 1) * NX * NY;
        cudaMemcpy(devices[device].data_in, h_A + device * slice_width * NX * NY, devices[device].number_values * sizeof(float), cudaMemcpyHostToDevice);
        //initialize halos
        //has left neighbor
        if(device != 0){
            cudaMalloc(&devices[device].data_halo_left_in, NX * NY * sizeof(float));
            cudaMalloc(&devices[device].data_halo_left_out, NX * NY * sizeof(float));
            cudaMemcpy(devices[device].data_halo_left_in, h_A + (device * slice_width - 1) * NX * NY, NX * NY * sizeof(float), cudaMemcpyHostToDevice);
        }
        //has right neighbor
        if(device != deviceCount -1){
            cudaMalloc(&devices[device].data_halo_right_in, NX * NY * sizeof(float));
            cudaMalloc(&devices[device].data_halo_right_out, NX * NY * sizeof(float));
            cudaMemcpy(devices[device].data_halo_right_in, h_A + (device + 1) * slice_width * NX * NY, NX * NY * sizeof(float), cudaMemcpyHostToDevice);
        }
    }

    float* h_buffer = (float *)malloc(NX * NY * sizeof(float)); //for copying data from device to host to device
    for(int i = 0; i<TSTEPS; i++){
        //let GPUs compute
        for(int device = 0; device < deviceCount; device++){
            cudaSetDevice(device);
            //COMPUTE
            int blocks = (devices[device].number_values + THREADS - 1) / THREADS;
            jacobi3D_kernel<<<blocks, THREADS>>>(devices[device], NX, NY, slice_width + (device == deviceCount-1) * (NZ % deviceCount), device, deviceCount);
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
            if(device != deviceCount - 1){
                cudaSetDevice(device+1);
                cudaMemcpy(h_buffer, devices[device+1].data_boundary_left_out, NX * NY * sizeof(float), cudaMemcpyDeviceToHost);
                cudaSetDevice(device);
                cudaMemcpy(devices[device].data_halo_right_in, h_buffer, NX * NY * sizeof(float), cudaMemcpyHostToDevice);
            }
            //Swap pointers
            if(i != TSTEPS -1){
                float* temp;
                temp = devices[device].data_in;
                devices[device].data_in = devices[device].data_out;
                devices[device].data_out = temp;

                temp = devices[device].data_interior_in;
                devices[device].data_interior_in = devices[device].data_interior_out;
                devices[device].data_interior_out = temp;

                temp = devices[device].data_boundary_left_in;
                devices[device].data_boundary_left_in = devices[device].data_boundary_left_out;
                devices[device].data_boundary_left_out = temp;

                temp = devices[device].data_boundary_right_in;
                devices[device].data_boundary_right_in = devices[device].data_boundary_right_out;
                devices[device].data_boundary_right_out = temp;

                //don't swap halo pointers because the data was updated through the host
            }
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
    int TSTEPS = 1000;
    unsigned int N = NX * NY * NZ;
    float *h_A = (float*)malloc(N * sizeof(float));

    // Initialize A. Layout A: [x=0, y=0, z=0], [x=1, y=0, z=0], ..., [x=NX-1, y=0, z=0], [x=0, y=1, z=0], ..., [x=NX-1, y=NY-1, z=0], [x=0, y=0, z=1], [x=1, y=0, z=1], ..., [x=NX-1, y=NY-1, z=NZ-1]
    for (int i = 0; i < N; i++) {
        h_A[i] = (float) i / N;
    }

    jacobi3D(h_A, NX, NY, NZ, TSTEPS);

    printf("Some values:\n");
    for(int i = 0; i < N; i += N/10){
            printf("%f ", h_A[i]);
    }
    printf("\n ");
    free(h_A);

    return 0;
}

#include <iostream>
#include<sys/time.h>

#include "NvInfer.h"
#include "NvInferRuntimeCommon.h"
#include "cuda_runtime_api.h"

#include "NvOnnxConfig.h"
#include "NvOnnxParser.h"

#include <cstdlib>

using namespace nvinfer1;
using namespace std;

//const char* onnx_file = "../model/mobilenet-v3.onnx";

class Logger:public ILogger{
	void log(Severity severity, const char* msg) override{
		if(severity == Severity::kERROR){
			std::cout << msg << std::endl;
		}
	}
} gLogger;


int build_infer(const char* onnx_file, IExecutionContext** context_, Dims* mInputDims, Dims* mOutputDims){
	// create
	IBuilder* builder = createInferBuilder(gLogger);

	const auto explicitBatch = 1U << static_cast<uint32_t>(NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);
	INetworkDefinition* network = builder->createNetworkV2(explicitBatch);

	IBuilderConfig* config = builder->createBuilderConfig();

	nvonnxparser::IParser* parser = nvonnxparser::createParser(*network, gLogger);

	//IOptimizationProfile* profile = builder->createOptimizationProfile();
	// config
	parser->parseFromFile(onnx_file, (int)ILogger::Severity::kWARNING);

	builder->setMaxBatchSize(1);
	config->setMaxWorkspaceSize(1 << 20);
	//profile->setDimensions("input", OptProfileSelector::kOPT, Dims3(3,32,32));
	// create engine
	ICudaEngine* engine = builder->buildEngineWithConfig(*network, *config);
	*mInputDims = network->getInput(0)->getDimensions();
	*mOutputDims = network->getOutput(0)->getDimensions();

	std::cout << "==== Network Info ====" << std::endl;
	std::cout << "Input: " << mInputDims->d[0] << ", " << mInputDims->d[1] << ", " <<mInputDims->d[2] << ", " << \
	mInputDims->d[3] << std::endl;
	std::cout << "Output: " << mOutputDims->d[0] << ", " << mOutputDims->d[1] << ", " <<mOutputDims->d[2] << ", " << \
	mOutputDims->d[3] << std::endl;

	IExecutionContext* context = engine->createExecutionContext();
	*context_ = context;
	return 0;
}

int start_infer(IExecutionContext* context, Dims& input_size, Dims& output_size){
	// malloc space
	float* input, *output;
	float* d_input, *d_output;

	int frame_size = input_size.d[1] * input_size.d[2] * input_size.d[3];
	input = (float*)malloc(sizeof(float)*frame_size);
	for(int i = 0; i < frame_size; i++){
		input[i] = 1; //(float)rand() / RAND_MAX;
	}
	cudaMalloc((void**)&d_input, sizeof(float)*frame_size);
	cudaMemcpy(d_input, input, sizeof(float)*frame_size, cudaMemcpyHostToDevice);

	int out_size = 10; //output_size.d[1];
	output = (float*)malloc(sizeof(float)*out_size);
	cudaMalloc((void**)&d_output, sizeof(float)*out_size);

	void *buffers[2];
	buffers[0] = (void*)d_input;
	buffers[1] = (void*)d_output;
	struct  timeval start, end;
	gettimeofday(&start, NULL);
	for(int i = 0; i < 1000; i++){
		context->executeV2(buffers);
	}
	gettimeofday(&end, NULL);
	float time = (end.tv_sec - start.tv_sec) * 1000 + (end.tv_usec - start.tv_usec)/1000;
	printf("Time cost: %f us\n", time);

	cudaMemcpy(output, d_output, sizeof(float)*out_size, cudaMemcpyDeviceToHost);
	for(int i = 0; i < out_size; i++){
		printf("%f ", output[i]);
	}
	printf("\n");
}

int main(int argc, char** argv){
	if(argc != 2){
		printf("Input onnx model\n");
		return -1;
	}
	char* onnx_file = argv[1];
	Dims input_size, output_size;
    IExecutionContext* context = nullptr;
	build_infer(onnx_file, &context, &input_size, &output_size);
	start_infer(context, input_size, output_size);
	return 0;
}

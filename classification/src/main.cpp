#include <iostream>
#include <NvInfer.h>
#include <NvInferRuntimeCommon.h>
#include <NvOnnxConfig.h>
#include <NvOnnxParser.h>

#include <cuda_runtime_api.h>
#include <unistd.h>

using namespace std;
using namespace nvinfer1;

const int MAXBATCHSIZE = 4;
const char* INPUT_BLOB_NAME = "input";
const char* OUTPUT_BLOB_NAME = "output";
const char* ONNX_FILE = "../model/sample.onnx";
const char* image_name = "dog.jpg";

class Logger:public ILogger {
	void log(Severity severity, const char *msg) override {
		if(severity == Severity::kINFO){
			std::cout << msg << std::endl;
		}
	}
} gLogger;

int network_def(const char* onnx_file, IBuilder** network_builder, INetworkDefinition** def_network, Dims* InputDims, Dims* OutputDims){
	// create the builder and network
	IBuilder *builder = createInferBuilder(gLogger);
    if(builder == nullptr){
		std::cerr << "create Infer Builder error" << endl;
		return -1;
    }
    const auto explictBatch = 1U << static_cast<uint32_t>(NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);
    INetworkDefinition* network = builder->createNetworkV2(explictBatch);
	*def_network = network;	

	// create the onnx parser
	nvonnxparser::IParser* parser = nvonnxparser::createParser(*network, gLogger);
    // load the model
	parser->parseFromFile(onnx_file, int(ILogger::Severity::kWARNING));
	
	*InputDims = network->getInput(0)->getDimensions();
	*OutputDims = network->getOutput(0)->getDimensions();
	cout << "InputDims: " << InputDims->d[0] << ", " << InputDims->d[1] << ", " << InputDims->d[2] << ", " << InputDims->d[3] << endl;
	cout << "OutputDims: " << OutputDims->d[0] << ", " << OutputDims->d[1] << ", " << OutputDims->d[2] << ", " << OutputDims->d[3] << endl;
    *network_builder = builder;
	return 0;
}

int engine_build(IBuilder* builder, INetworkDefinition* network, ICudaEngine** network_engine){
	builder->setMaxBatchSize(MAXBATCHSIZE);
	IBuilderConfig* config = builder->createBuilderConfig();
	config->setMaxWorkspaceSize(1 << 20);
    ICudaEngine* engine = builder->buildEngineWithConfig(*network, *config);

	*network_engine = engine;
	return 0;
}

int prepare_inference(ICudaEngine* engine, IExecutionContext ** network_context, int* inputIndex, int* outputIndex){
	IExecutionContext *context = engine->createExecutionContext();

    *inputIndex = engine->getBindingIndex(INPUT_BLOB_NAME);
    *outputIndex = engine->getBindingIndex(OUTPUT_BLOB_NAME);
	int num = engine->getNbBindings();
	std::cout << "Input: " << *inputIndex << ", Output: " << *outputIndex << ", Num: " << num << endl;
   
	*network_context = context;
	return 0;
}

int image_file_read(const char* image_name, Dims& dims, void** image_buffer){
	int size = dims.d[1] * dims.d[2] * dims.d[3];
    float* image = new float[size];
	for(int i = 0; i < size; i++){
		image[i] = 1; ///(i+1);
	}
    // cuda operate
	void* buffer = nullptr;
	cudaMalloc(&buffer, size*sizeof(float));
    cudaMemcpy(&buffer, image, size*sizeof(float), cudaMemcpyHostToDevice);
	*image_buffer = buffer;

    return 0;
}

int do_inference(IExecutionContext* context, Dims& outputDims, void* image_buffer, int& inputIndex, int& outputIndex, void** out_buffer){
	int size = 1; //outputDims.d[1];
    void* res_buffer = nullptr;
	//cudaMalloc(&res_buffer, size*sizeof(float));
	cudaError_t ret = cudaMalloc(&res_buffer, size*sizeof(float));
    if(ret != cudaSuccess){
		cerr << "cuda Malloc error, errorcode: " << ret << endl;
	}
	
	//void *buffer[2];
	//buffer[0] = image_buffer;
    //buffer[1] = res_buffer;
	//context->executeV2(buffer);
   	std::vector<void*> buffer(2);
	buffer[0] = image_buffer;
	buffer[1] = res_buffer; 
	context->executeV2(buffer.data());

	usleep(10*1000);

	float *output_buffer = new float[size];
    cudaMemcpy(output_buffer, res_buffer, size*sizeof(float), cudaMemcpyDeviceToHost);
	*out_buffer = output_buffer;
    cout << "do inference finish" << endl;
	printf("Result: %f\n", *output_buffer);
    return 0;
}

int inference_res_show(void* output_buffer, Dims& outputDims){
	int size = 1; //outputDims.d[1];
    float *buffer = (float*)output_buffer;
	for(int i = 0; i < size; i++){
		cout << buffer[i] << ", ";
	}
	cout << endl;
	return 0;
}

int main(){
	IBuilder *builder = nullptr;
	INetworkDefinition *network = nullptr;
	ICudaEngine *engine = nullptr;
	IExecutionContext* context = nullptr;

	Dims inputDims, outputDims;
	void* input_buffer = nullptr;
	void* output_buffer = nullptr;
    int inputIndex, outputIndex;

	network_def(ONNX_FILE, &builder, &network, &inputDims, &outputDims);
    engine_build(builder, network, &engine);
	prepare_inference(engine, &context, &inputIndex, &outputIndex);

	image_file_read(image_name, inputDims, &input_buffer);
	do_inference(context, outputDims, input_buffer, inputIndex, outputIndex, &output_buffer);
    inference_res_show(output_buffer, outputDims);
	
	return 0;
}

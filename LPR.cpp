#include <opencv2/highgui.hpp>
#include <iostream>

#include "tensorflow/cc/ops/const_op.h"
#include "tensorflow/cc/ops/image_ops.h"
#include "tensorflow/cc/ops/standard_ops.h"
#include "tensorflow/core/framework/graph.pb.h"
#include "tensorflow/core/graph/default_device.h"
#include "tensorflow/core/graph/graph_def_builder.h"
#include "tensorflow/core/lib/core/threadpool.h"
#include "tensorflow/core/lib/io/path.h"
#include "tensorflow/core/lib/strings/stringprintf.h"
#include "tensorflow/core/platform/init_main.h"
#include "tensorflow/core/public/session.h"
#include "tensorflow/core/util/command_line_flags.h"
 
#include "utils.h"

using tensorflow::Flag;
using tensorflow::Tensor;
using tensorflow::Status;
using tensorflow::string;
using tensorflow::int32;

using namespace std;
using namespace cv;

 
int main( int argc, char** argv ) {
  
  string LABELS = "Train3/PLATE_LABEL.pbtxt";
  string GRAPH = "inference_graph_3/frozen_inference_graph.pb";

  // Set input & output nodes names
  string inputLayer = "image_tensor:0";
  vector<string> outputLayer = {"detection_boxes:0", "detection_scores:0", "detection_classes:0", "num_detections:0"};
  
  // Load and initialize the model from .pb file
  std::unique_ptr<tensorflow::Session> session;
  string graphPath = tensorflow::io::JoinPath(ROOTDIR, GRAPH);
  LOG(INFO) << "graphPath:" << graphPath;
  Status loadGraphStatus = loadGraph(graphPath, &session);
  if (!loadGraphStatus.ok()) {
        LOG(ERROR) << "loadGraph(): ERROR" << loadGraphStatus;
        return -1;
    } else
  LOG(INFO) << "loadGraph(): frozen graph loaded" << endl;


  // Load labels map from .pbtxt file
  std::map<int, std::string> labelsMap = std::map<int,std::string>();
  Status readLabelsMapStatus = readLabelsMapFile(tensorflow::io::JoinPath(ROOTDIR, LABELS), labelsMap);
  if (!readLabelsMapStatus.ok()) {
        LOG(ERROR) << "readLabelsMapFile(): ERROR" << loadGraphStatus;
        return -1;
    } else
  LOG(INFO) << "readLabelsMapFile(): labels map loaded with " << labelsMap.size() << " label(s)" << endl;

  cv::Mat image;
  Tensor tensor;
  std::vector<Tensor> outputs;
  double thresholdScore = 0.5;
  double thresholdIOU = 0.8;
  
  image = cv::imread(argv[1], CV_LOAD_IMAGE_COLOR);
  
  if(! image.data ) {
      std::cout <<  "Could not open or find the image" << std::endl ;
      return -1;
    }
  

  // Convert mat to tensor
  tensor = Tensor(tensorflow::DT_FLOAT, shape);
  Status readTensorStatus = readTensorFromMat(image, tensor);
  if (!readTensorStatus.ok()) {
     LOG(ERROR) << "Mat->Tensor conversion failed: " << readTensorStatus;
     return -1;
          }

 // Run the graph on tensor
  outputs.clear();
  Status runStatus = session->Run({{inputLayer, tensor}}, outputLayer, {}, &outputs);
  if (!runStatus.ok()) {
      LOG(ERROR) << "Running model failed: " << runStatus;
      return -1;
      }

  // Extract results from the outputs vector
  tensorflow::TTypes<float>::Flat scores = outputs[1].flat<float>();
  tensorflow::TTypes<float>::Flat classes = outputs[2].flat<float>();
  tensorflow::TTypes<float>::Flat numDetections = outputs[3].flat<float>();
  tensorflow::TTypes<float, 3>::Tensor boxes = outputs[0].flat_outer_dims<float,3>();

  vector<size_t> goodIdxs = filterBoxes(scores, boxes, thresholdIOU, thresholdScore);
  for (size_t i = 0; i < goodIdxs.size(); i++)
      LOG(INFO) << "score:" << scores(goodIdxs.at(i)) << ",class:" << labelsMap[classes(goodIdxs.at(i))]
                << " (" << classes(goodIdxs.at(i)) << "), box:" << "," << boxes(0, goodIdxs.at(i), 0) << ","
                << boxes(0, goodIdxs.at(i), 1) << "," << boxes(0, goodIdxs.at(i), 2) << ","
                << boxes(0, goodIdxs.at(i), 3);
  
  // Draw bboxes and captions
  cvtColor(image, COLOR_BGR2RGB);
  drawBoundingBoxesOnImage(image, scores, classes, boxes, labelsMap, goodIdxs);

  cv::namedWindow( "Display window", cv::WINDOW_AUTOSIZE );
  cv::imshow( "Display window", image );
  
  cv::waitKey(0);
  return 0;
}

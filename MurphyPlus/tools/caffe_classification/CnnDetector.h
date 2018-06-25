#ifndef __CNN_DETECTOR_H__
#define __CNN_DETECTOR_H__

using namespace caffe;  // NOLINT(build/namespaces)
using std::string;
using namespace cv;
using namespace std; 
/* Pair (label, confidence) representing a prediction. */
typedef std::pair<string, float> Prediction;

class Classifier {
public:
	Classifier(const string& model_file,
	const string& trained_file,
	const string& mean_file,
	const string& label_file);

	std::vector<Prediction> Classify(const cv::Mat& img, int N = 5);
	std::vector<float> Predict(const cv::Mat& img);

private:
	void SetMean(const string& mean_file);
	void WrapInputLayer(std::vector<cv::Mat>* input_channels);

	void Preprocess(const cv::Mat& img,
	std::vector<cv::Mat>* input_channels);

private:
	shared_ptr<Net<float> > net_;
	cv::Size input_geometry_;
	int num_channels_;
	cv::Mat mean_;
	std::vector<string> labels_;
};

extern void *CnnDetectorInit(char *model_file,char *trained_file,char *mean_file,char *label_file);
extern int CnnDetectorPredict(Mat &img,vector<float> &out);



#endif
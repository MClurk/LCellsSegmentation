/*  Zang Hui
 *  huezann@gmail.com
 *  superpixel pre-processing + GraphCut image segmentation
 *  2013 - 2014
 */
#include <vector>
#include <string>
#include <opencv/cv.h>
#include <opencv/cxcore.h>
#include <opencv/highgui.h>
using namespace std;

typedef cv::Scalar_<unsigned char> Color;
// 命名空间还没有写呢!!!
//预编译头！！！
//最后把它移出去试试看
class LcellsGraphcutSegmentation {
public:
	typedef cv::Mat Mat;

	enum {
		PIXEL_TYPE_UNKNOW = 0,
		PIXEL_TYPE_BKG = 1,
		PIXEL_TYPE_OBJ = 2
	};
	struct SuperpixelColor {
		SuperpixelColor():vec_histogram(4096, 0) {}
		vector<double> vec_histogram;
	};
	struct Similarity
	{
		Similarity(unsigned int label1 = 0, unsigned int label2 = 0, double similarity = 0):
			int_label1(label1), int_label2(label2), dou_similarity(similarity) {}
		unsigned int int_label1;
		unsigned int int_label2;
		double dou_similarity;
	};

	LcellsGraphcutSegmentation();
	LcellsGraphcutSegmentation(const Mat& src_image, const string filename, const double dou_lambda);
	~LcellsGraphcutSegmentation();
	bool runSegment(const Mat& seed_image);

private:
	const Mat src_image;
	const string label_filename;
	const unsigned int rows, cols;
	double dou_lambda;
	unsigned int int_superpixel_number;
	vector< vector<unsigned int> > vec2D_superpixel_label;
	vector<SuperpixelColor> vec_image_superpixel_color;
	vector<unsigned int> vec_pixel_number_each_superpixel;

	vector<unsigned int> vec_obj_labels;
	vector<unsigned int> vec_bkg_labels;
	vector<unsigned char> vec_superpixel_type;

	LcellsGraphcutSegmentation(const LcellsGraphcutSegmentation& );  //not allow copy and assign
	LcellsGraphcutSegmentation& operator = (const LcellsGraphcutSegmentation& );

protected:
	bool readTheSuperpixelLabel();
	bool calcColorEachSuperpixel();
	bool calcSeeds(const Mat& seed_image);
	bool initialize(const Mat& seed_image);
	double calcMostSimilar(const SuperpixelColor& superpixel_color,
						   const vector<unsigned int>& vec_obj_or_bkg);
	bool calcNLink(vector<Similarity>& vec_nlink_similarity);
	bool calcTLink(vector<double>& vec_tlink_obj_similarity, vector<double>& vec_tlink_bkg_similarity);
	bool runGraphCut(const vector<Similarity>& vec_nlink_similarity,
				  	  	  	  	  	  	  	  const vector<double>& vec_tlink_obj_similarity,
				  	  	  	  	  	  	  	  const vector<double>& vec_tlink_bkg_similarity);
	bool extractObject();

};  //class LcellsGraphcutSegmentation

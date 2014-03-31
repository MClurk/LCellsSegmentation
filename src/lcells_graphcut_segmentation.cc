/*  Zang Hui
 *  huezann@gmail.com
 *  superpixel pre-processing + GraphCut image segmentation
 *  2013 - 2014
 */
#include "lcells_graphcut_segmentation.h"
#include <cassert>
#include <cfloat>
#include <cstdio>
#include <iostream>
#include <algorithm>
#include "maxflow-v3.01/graph.h"
using namespace cv;
using namespace std;

LcellsGraphcutSegmentation::LcellsGraphcutSegmentation(): rows(0),
														  cols(0),
														  dou_lambda(0),
														  int_superpixel_number(0) {}

LcellsGraphcutSegmentation::LcellsGraphcutSegmentation(const Mat& src_image,
													   const string filename,
													   const double dou_lambda):
															   src_image(src_image)/*here is no copy of data*/,
															   label_filename(filename),
															   rows(src_image.rows),
															   cols(src_image.cols),
															   dou_lambda(dou_lambda),
															   int_superpixel_number(0) {}
LcellsGraphcutSegmentation::~LcellsGraphcutSegmentation() {}
bool LcellsGraphcutSegmentation::readTheSuperpixelLabel() {
	vector<unsigned int> vec_label_row(cols, 0);
	vec2D_superpixel_label.assign(rows, vec_label_row);
	FILE* ptr_label_txt = fopen(label_filename.c_str(), "r");
	if (!ptr_label_txt)
		return false;
	int label = 0;
	unsigned int count = 0;
	while( count < rows * cols ) {
		fscanf(ptr_label_txt, "%d", &label);
		vec2D_superpixel_label[floor((double)count / cols)][count % cols] = label;
		++count;
		}
	fclose (ptr_label_txt);
	return true;
}

bool LcellsGraphcutSegmentation::calcColorEachSuperpixel() {
	for(unsigned int y = 0; y < rows; ++y) {
		for(unsigned int x = 0; x < cols; ++x) {
			if(vec2D_superpixel_label[y][x] > int_superpixel_number)
				int_superpixel_number = vec2D_superpixel_label[y][x];
		}
	}
	int_superpixel_number += 1;

	vec_image_superpixel_color.assign(int_superpixel_number, SuperpixelColor());
	vec_pixel_number_each_superpixel.assign(int_superpixel_number, 0);  //for normalize use
	/*if ( src_image.isContinuous() ) {
		cols *= rows;
		rows = 1;
	}*/
	for (unsigned int y = 0; y < rows; ++y)
	{
		const unsigned char* ptr_data = src_image.ptr(y);
		for (unsigned int x = 0; x < cols; ++x) {
			unsigned int int_label = vec2D_superpixel_label[y][x];  //label start from 0
			unsigned int/*cannot be uchar*/ int_color = (unsigned int)(ptr_data[3 * x + 0] / 16) * 16 * 16 +
														(unsigned int)(ptr_data[3 * x + 1] / 16) * 16 +
														(unsigned int)(ptr_data[3 * x + 2] / 16);
			vec_image_superpixel_color[int_label].vec_histogram[int_color] += 1;  //index from 0 to 4095
			vec_pixel_number_each_superpixel[int_label] += 1;
		}
	}
	for (unsigned int i = 0; i < int_superpixel_number; ++i) {
		SuperpixelColor &this_superpixel_color = vec_image_superpixel_color[i];
		unsigned int int_pixel_number = vec_pixel_number_each_superpixel[i];
		for (int k = 0; k < 4096; ++k) {
			if (0 == int_pixel_number)
				continue;  //if there is a blank label, let it blank
			this_superpixel_color.vec_histogram[k] =
					this_superpixel_color.vec_histogram[k] /
					(double)int_pixel_number;
		}
	}
	return true;
}

bool LcellsGraphcutSegmentation::calcSeeds(const Mat& seed_image) {
	if( seed_image.empty() || 1 != seed_image.channels() )
		return false;
 	if(vec2D_superpixel_label.empty())
		return false;

	vec_superpixel_type.assign(int_superpixel_number, PIXEL_TYPE_UNKNOW);
	for(unsigned int y = 0; y < rows; ++y)
	{
		const unsigned char* ptr_data = seed_image.ptr(y);
		for(unsigned int x = 0; x < cols; ++x)
		{
			unsigned char uchar_seed_color = ptr_data[x];
			if (0 == uchar_seed_color) {}
			else if(255 == uchar_seed_color) {
				unsigned int int_label = vec2D_superpixel_label[y][x];
				vec_superpixel_type[int_label] = PIXEL_TYPE_OBJ;
				vec_obj_labels.push_back(int_label);
			}  //object
			else if(128 == uchar_seed_color) {
				int int_label = vec2D_superpixel_label[y][x];
				vec_superpixel_type[int_label] = PIXEL_TYPE_BKG;
				vec_bkg_labels.push_back(int_label);
			}  //background
		}
	}
	return true;
}

bool LcellsGraphcutSegmentation::initialize(const Mat& seed_image) {
	if ( readTheSuperpixelLabel() )
		if ( calcColorEachSuperpixel() )
			if ( calcSeeds(seed_image) )
				return true;
	return false;
}

double LcellsGraphcutSegmentation::calcMostSimilar(const SuperpixelColor& superpixel_color,
												   const vector<unsigned int>& vec_seed_label) {
	//assert(!vec_colors.empty());
	double dou_max_similar = 0;  //0 is the min value
	vector<unsigned int>::const_iterator iter_label = vec_seed_label.begin();
	for (/*iter_label*/; iter_label != vec_seed_label.end(); ++iter_label) {
		double dou_similar = 0;
		SuperpixelColor &seed_superpixel_color = vec_image_superpixel_color[*iter_label];
		for(unsigned int k = 0; k < 4096; ++k) {
			dou_similar += sqrt(superpixel_color.vec_histogram[k] * seed_superpixel_color.vec_histogram[k]);
		}
		if(dou_similar > dou_max_similar)
			dou_max_similar = dou_similar;
	}
	return dou_max_similar;
}

bool LcellsGraphcutSegmentation::calcNLink(vector<Similarity>& vec_nlink_similarity) {
	if (vec2D_superpixel_label.empty())
		return false;
	if (vec_superpixel_type.empty())
		return false;

	vector<bool> vec_row(int_superpixel_number, false);
	vector< vector<bool> > vec2D_edge_visited(int_superpixel_number, vec_row);  //to find the edges
	for (unsigned int y = 0; y < rows; ++y) {
		int int_minY = std::max((int)y - 1, 0);  //dont cast them to unsigned(_Tp)!
		int int_maxY = std::min(y + 1, rows - 1);
		for (unsigned int x = 0; x < cols; ++x) {
			unsigned int int_label = vec2D_superpixel_label[y][x];
			int int_minX = std::max((int)x - 1, 0);
			int int_maxX = std::min(x + 1, cols - 1);
			for (int nei_Y = int_minY; nei_Y <= int_maxY; ++nei_Y) {
				for (int nei_X = int_minX; nei_X <= int_maxX; ++nei_X) {
					unsigned int int_nei_label = vec2D_superpixel_label[nei_Y][nei_X];
					if (int_label != int_nei_label) {
						unsigned int label1 = std::min(int_label, int_nei_label);
						unsigned int label2 = std::max(int_label, int_nei_label);
						if (false == vec2D_edge_visited[label1][label2]) {
							vec2D_edge_visited[label1][label2] = true;
							vec_nlink_similarity.push_back(Similarity(label1, label2, 0));
						}
					}
				}
			}
		}
	}

	unsigned int int_edge_number = vec_nlink_similarity.size();
	for (unsigned int index = 0; index < int_edge_number; ++index) {
		unsigned int left_label = vec_nlink_similarity[index].int_label1;
		unsigned int right_label = vec_nlink_similarity[index].int_label2;
		double similarity = 0;
		SuperpixelColor &left_color = vec_image_superpixel_color[left_label];
		SuperpixelColor &right_color = vec_image_superpixel_color[right_label];
		for(unsigned int k = 0; k < 4096; ++k) {
			similarity += sqrt(left_color.vec_histogram[k] *
						  	   right_color.vec_histogram[k]);
		}
		vec_nlink_similarity[index].dou_similarity = dou_lambda * similarity;
	}
	return true;
}

bool LcellsGraphcutSegmentation::calcTLink(vector<double>& vec_tlink_obj_similarity,
			   	   	   	   	   	   	   	   vector<double>& vec_tlink_bkg_similarity) {
	if(vec_superpixel_type.empty())
		return false;
	if(vec_obj_labels.empty() || vec_bkg_labels.empty())
		return false;

	vec_tlink_obj_similarity.assign(int_superpixel_number, 0);
	vec_tlink_bkg_similarity.assign(int_superpixel_number, 0);
	for (unsigned int i = 0; i < int_superpixel_number; ++i) {
		double dou_obj_dist, dou_bkg_dist;
		if (0 == vec_pixel_number_each_superpixel[i])
			continue;  //not exist label
		switch (vec_superpixel_type[i]) {
		case PIXEL_TYPE_UNKNOW:
			dou_obj_dist = calcMostSimilar(vec_image_superpixel_color[i], vec_obj_labels);
			dou_bkg_dist = calcMostSimilar(vec_image_superpixel_color[i], vec_bkg_labels);
			vec_tlink_obj_similarity[i] = dou_obj_dist / (dou_obj_dist + dou_bkg_dist);
			vec_tlink_bkg_similarity[i] = dou_bkg_dist / (dou_obj_dist + dou_bkg_dist);
			break;
		case PIXEL_TYPE_BKG:
			vec_tlink_obj_similarity[i] = 0.;
			vec_tlink_bkg_similarity[i] = 5.;
			break;
		case PIXEL_TYPE_OBJ:
			vec_tlink_bkg_similarity[i] = 0.;
			vec_tlink_obj_similarity[i] = 5.;
			break;
		default:
			return false;
		}
	}
	return true;
}

bool LcellsGraphcutSegmentation::runGraphCut(const vector<Similarity>& vec_nlink_similarity,
			  	  	  	  	  	  	  	  	 const vector<double>& vec_tlink_obj_similarity,
			  	  	  	  	  	  	  	     const vector<double>& vec_tlink_bkg_similarity) {
	if (vec_superpixel_type.empty())
		return false;
	if (vec_nlink_similarity.empty())
		return false;
	if (vec_tlink_obj_similarity.empty() || vec_tlink_bkg_similarity.empty())
		return false;

	typedef Graph<double,double,double> GraphType;
	GraphType *graph = new GraphType(0, 0);
	graph->add_node(int_superpixel_number);  //include the not exist label

	unsigned int int_edge_number = vec_nlink_similarity.size();
	for (unsigned int i = 0; i < int_edge_number; i++) {
		int left_node = vec_nlink_similarity[i].int_label1;
		int right_node = vec_nlink_similarity[i].int_label2;
		graph->add_edge(left_node, right_node,
						vec_nlink_similarity[i].dou_similarity,
						vec_nlink_similarity[i].dou_similarity);
	}
	for (unsigned int i = 0; i < int_superpixel_number; i++) {
		graph->add_tweights(i, vec_tlink_obj_similarity[i], vec_tlink_bkg_similarity[i]);
	}
	graph->maxflow();
	for (unsigned int i = 0; i < int_superpixel_number; i++) {
		if(GraphType::SOURCE == graph->what_segment(i))
			vec_superpixel_type[i] = PIXEL_TYPE_OBJ; //foreground
		else
			vec_superpixel_type[i] = PIXEL_TYPE_BKG; //background
	}
	delete graph;
	return true;
}

bool LcellsGraphcutSegmentation::extractObject() {

	Mat segment_result_image = src_image.clone();
	Mat segment_result_binary_image(rows, cols, CV_8UC1);
	segment_result_binary_image = Color::all(0);

	double dou_alpha = 0.5;
	Color color((1. - dou_alpha) * 255, (1. - dou_alpha) * 0, (1. - dou_alpha) * 128);
	for (unsigned int y = 0; y < rows; ++y) {
		unsigned char* ptr_data_result = segment_result_image.ptr(y);
		unsigned char* ptr_data_binary = segment_result_binary_image.ptr(y);
		for (unsigned int x = 0; x < cols; ++x) {
			switch ( vec_superpixel_type[vec2D_superpixel_label[y][x]] ) {
			case PIXEL_TYPE_OBJ:
				ptr_data_binary[x] = 255;
				break;
			case PIXEL_TYPE_BKG:
				ptr_data_result[3 * x + 0] = ptr_data_result[3 * x + 0] * dou_alpha + color.val[0];
				ptr_data_result[3 * x + 1] = ptr_data_result[3 * x + 1] * dou_alpha + color.val[1];
				ptr_data_result[3 * x + 2] = ptr_data_result[3 * x + 2] * dou_alpha + color.val[2];
				break;
			}
		}
	}
	imshow("binary result", segment_result_binary_image);
	imshow("result segmentation", segment_result_image);
	return true;
}

bool LcellsGraphcutSegmentation::runSegment(const Mat& seed_image) {
	if ( src_image.empty() || 3 != src_image.channels() )
		return false;
	if ( seed_image.empty() || 1 != seed_image.channels() )
		return false;

	bool bool_return_value = false;
	bool_return_value = initialize(seed_image);
	if (!bool_return_value)
		return false;

	vector<Similarity> vec_nlink_similarity;
	vector<double> vec_tlink_obj_similarity, vec_tlink_bkg_similarity;
	if (!calcNLink(vec_nlink_similarity) || !calcTLink(vec_tlink_obj_similarity, vec_tlink_bkg_similarity))
		return false;

	bool_return_value = runGraphCut(vec_nlink_similarity, vec_tlink_obj_similarity, vec_tlink_bkg_similarity);
	if (!bool_return_value)
		return false;

	bool_return_value = extractObject();
	if (!bool_return_value)
		return false;
	return true;
}

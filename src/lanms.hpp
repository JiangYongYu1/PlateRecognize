#pragma once

#include "clipper.hpp"
#include<numeric>
#include "assert.h"
#include<limits>
#include<algorithm>

// locality-aware NMS
namespace lanms {

	namespace cl = ClipperLib;

	struct Polygon {
		cl::Path poly;
		float score;
		float poly_sin;
		float poly_cos;
	};

	float paths_area(const ClipperLib::Paths &ps);

	float poly_iou(const Polygon &a, const Polygon &b);

	bool should_merge(const Polygon &a, const Polygon &b, float iou_threshold);

	/**
	 * Incrementally merge polygons
	 */
	class PolyMerger {
		public:
			PolyMerger();

			/**
			 * Add a new polygon to be merged.
			 */
			void add(const Polygon &p_given);

			inline std::int64_t sqr(std::int64_t x);

			Polygon normalize_poly(
					const Polygon &ref,
					const Polygon &p);

			Polygon get() const;

		private:
			std::int64_t data[8];
			float score;
			float poly_sin;
			float poly_cos;
			std::int32_t nr_polys;
	};


	/**
	 * The standard NMS algorithm.
	 */
	std::vector<Polygon> standard_nms(std::vector<Polygon> &polys, float iou_threshold);

	std::vector<Polygon>
		merge_quadrangle_n9(const float *data, size_t n, float iou_threshold);
}


namespace lanms_adaptor {

	std::vector<std::vector<float>> polys2floats(const std::vector<lanms::Polygon> &polys);
	/**
	 *
	 * \param quad_n11 an n-by-11 numpy array, where first 8 numbers denote the
	 *		quadrangle, and the 9 is the score
	 * \param iou_threshold two quadrangles with iou score above this threshold
	 *		will be merged
	 *
	 * \return an n-by-11 numpy array, the merged quadrangles
	 */
	std::vector<std::vector<float>> merge_quadrangle_n9(
			float* ptr, int n,
			float iou_threshold);
}
/**
 *  SIBIA (Scalable Biophysics-Based Image Analysis)
 *
 *  Copyright (C) 2017-2020, The University of Texas at Austin
 *  This file is part of the SIBIA library.
 *
 *  SIBIA is free software: you can redistribute it and/or modify
 *  it under the terms of the GNU General Public License as published by
 *  the Free Software Foundation, either version 3 of the License, or
 *  (at your option) any later version.
 *
 *  SIBIA is distributed in the hope that it will be useful,
 *  but WITHOUT ANY WARRANTY; without even the implied warranty of
 *  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 *  GNU General Public License for more details.
 *
 *  You should have received a copy of the GNU General Public License
 *  along with this program. If not, see the LICENSE file.
 *
 **/

#ifndef PARAMETERS_H_
#define PARAMETERS_H_

#include <stdlib.h>

struct OptimizerSettings {
  // TODO
};

struct TumorParameters {
  // TODO
};

struct Grid {
  int[3] n_;
  int nl_;
  int64_t ng_;
  // TODO
};


struct FilePaths {
  public:
    FilePaths() :
    wm_(), gm_(), csf_(), ve_(), glm_(), data_t1_(), data_t0_(),
    data_support_(), data_comps(), obs_filter_(),
    velocity_x_1(), velocity_x_2(), velocity_x_3(),
    pvec_()
    {}

    // material properties
    std::string wm_;
    std::string gm_;
    std::string csf_;
    std::string ve_;
    std::string glm_;
    // data
    std::string data_t1_;
    std::string data_t0_;
    std::string data_support_;
    std::string data_comps_;
    std::string obs_filter_;
    // velocity
    std::string velocity_x1_;
    std::string velocity_x2_;
    std::string velocity_x3_;
    // warmstart solution
    std::string pvec_;
    std::string phi_;
};

class Parameters {
  public:
    Parameters() :
      opt_(),
      tu_(),
      path_(),
      grid_() {

    opt_ = std::make_shared<OptimizerSettings>();
    tu_ = std::make_shared<TumorParameters>();
    path_ = std::make_shared<FilePaths>();
    grid_ = std::make_shared<Grid>();
    }

    virtual ~Parameters() {}

  private:
    std::shared_ptr<OptimizerSettings> opt_;
    std::shared_ptr<TumorParameters> tu_;
    std::shared_ptr<FilePaths> path_;
    std::shared_ptr<Grid> grid_;


    ScalarType obs_threshold_0_;
    ScalarType obs_threshold_1_;

    bool relative_obs_threshold_;
};

#endif

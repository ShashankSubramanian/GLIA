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
    data_support_(), data_support_data_(), data_comps(), data_comps_data_(),
    obs_filter_(), mri_(), velocity_x_1(), velocity_x_2(), velocity_x_3(),
    pvec_(), phi_(),
    writepath_(), readpath_()
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
    std::string data_support_data_;
    std::string data_comps_;
    std::string data_comps_data_;
    std::string obs_filter_;
    std::string mri_;
    // velocity
    std::string velocity_x1_;
    std::string velocity_x2_;
    std::string velocity_x3_;
    // warmstart solution
    std::string pvec_;
    std::string phi_;

    std::string writepath_;
    std::string readpath_;
};

class Parameters {
  public:
    Parameters() :
      obs_threshold_0_(-1),
      obs_threshold_1_(-1),
      pre_adv_time_(-1),
      opt_(),
      tu_(),
      path_(),
      grid_() {

    opt_ = std::make_shared<OptimizerSettings>();
    tu_ = std::make_shared<TumorParameters>();
    path_ = std::make_shared<FilePaths>();
    grid_ = std::make_shared<Grid>();
    }

    inline int get_nk() {return tu_->diffusivity_inversion_ ? params_->tu_->nk_ : 0;}
    inline int get_nr() {return tu_->reaction_inversion_ ? params_->tu_->nr_ : 0;}

    virtual ~Parameters() {}

  private:
    ScalarType obs_threshold_0_;
    ScalarType obs_threshold_1_;
    ScalarType pre_adv_time_;

    bool relative_obs_threshold_;
    bool inject_coarse_sol_;
    bool two_time_points_;

    int sparsity_level_;

    std::shared_ptr<OptimizerSettings> opt_;
    std::shared_ptr<TumorParameters> tu_;
    std::shared_ptr<FilePaths> path_;
    std::shared_ptr<Grid> grid_;
};

#endif

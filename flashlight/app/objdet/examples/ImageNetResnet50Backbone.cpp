/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <exception>
#include <iomanip>

#include <gflags/gflags.h>
#include <glog/logging.h>

#include "flashlight/fl/common/Filesystem.h"
#include "flashlight/fl/dataset/datasets.h"
#include "flashlight/fl/meter/meters.h"
#include "flashlight/fl/optim/optim.h"
#include "flashlight/fl/tensor/Compute.h"
#include "flashlight/fl/tensor/Init.h"
#include "flashlight/fl/tensor/Random.h"
#include "flashlight/pkg/runtime/common/DistributedUtils.h"
#include "flashlight/pkg/vision/dataset/DistributedDataset.h"
#include "flashlight/pkg/vision/dataset/Imagenet.h"
#include "flashlight/pkg/vision/dataset/Transforms.h"
#include "flashlight/pkg/vision/models/Resnet50Backbone.h"

DEFINE_string(data_dir, "", "Directory of imagenet data");
DEFINE_double(train_lr, 0.1f, "Learning rate");
DEFINE_double(train_momentum, 0.9f, "Momentum");

DEFINE_double(train_wd, 1e-4f, "Weight decay");
DEFINE_uint64(train_epochs, 50, "Number of epochs to train");
DEFINE_bool(distributed_enable, true, "Enable distributed training");
DEFINE_int64(
    distributed_max_devices_per_node,
    8,
    "the maximum number of devices per training node");
DEFINE_int64(
    distributed_world_rank,
    0,
    "rank of the process (Used if distributed_rndv_filepath is not empty)");
DEFINE_int64(
    distributed_world_size,
    1,
    "total number of the process (Used if distributed_rndv_filepath is not empty)");
DEFINE_string(
    distributed_rndv_filepath,
    "",
    "Shared file path used for setting up rendezvous."
    "If empty, uses MPI to initialize.");
DEFINE_uint64(data_batch_size, 256, "Total batch size across all gpus");
DEFINE_string(exp_checkpoint_path, "/tmp/model", "Checkpointing prefix path");
DEFINE_int64(exp_checkpoint_epoch, -1, "Checkpoint epoch to load from");

using namespace fl;
using fl::pkg::vision::compose;
using fl::pkg::vision::ImageTransform;
using namespace fl::pkg::vision;

#define FL_LOG_MASTER(lvl) LOG_IF(lvl, (fl::getWorldRank() == 0))

// Returns the average loss, top 5 error, and top 1 error
std::tuple<double, double, double> evalLoop(
    std::shared_ptr<Module> model,
    Dataset& dataset) {
  AverageValueMeter lossMeter;
  TopKMeter top5Acc(5);
  TopKMeter top1Acc(1);

  // Place the model in eval mode.
  model->eval();
  for (auto& example : dataset) {
    auto inputs = noGrad(example[kImagenetInputIdx]);
    auto output = model->forward({inputs})[0];

    auto target = noGrad(example[kImagenetTargetIdx]);

    // Compute and record the loss.
    auto loss = categoricalCrossEntropy(output, target);
    lossMeter.add(loss.tensor().scalar<float>());
    top5Acc.add(output.tensor(), target.tensor());
    top1Acc.add(output.tensor(), target.tensor());
  }
  model->train();
  fl::pkg::runtime::syncMeter(lossMeter);
  fl::pkg::runtime::syncMeter(top5Acc);
  fl::pkg::runtime::syncMeter(top1Acc);

  double loss = lossMeter.value()[0];
  return std::make_tuple(loss, top5Acc.value(), top1Acc.value());
};

int main(int argc, char** argv) {
  fl::init();
  google::InitGoogleLogging(argv[0]);
  google::InstallFailureSignalHandler();
  gflags::ParseCommandLineFlags(&argc, &argv, true);

  const std::string labelPath = fs::path(FLAGS_data_dir) / "labels.txt";
  const std::string trainList = fs::path(FLAGS_data_dir) / "train";
  const std::string valList = fs::path(FLAGS_data_dir) / "val";

  /////////////////////////
  // Setup distributed training
  ////////////////////////
  if (FLAGS_distributed_enable) {
    fl::pkg::runtime::initDistributed(
        FLAGS_distributed_world_rank,
        FLAGS_distributed_world_size,
        FLAGS_distributed_max_devices_per_node,
        FLAGS_distributed_rndv_filepath);
  }
  const int worldRank = fl::getWorldRank();
  const int worldSize = fl::getWorldSize();
  const bool isMaster = (worldRank == 0);

  fl::setDevice(worldRank);
  fl::setSeed(worldSize);

  auto reducer =
      std::make_shared<fl::CoalescingReducer>(1.0 / worldSize, true, true);

  //////////////////////////
  //  Create datasets
  /////////////////////////
  // These are the mean and std for each channel of Imagenet
  const std::vector<float> mean = {0.485, 0.456, 0.406};
  const std::vector<float> std = {0.229, 0.224, 0.225};
  const int randomResizeMax = 480;
  const int randomResizeMin = 256;
  const int randomCropSize = 224;
  const float horizontalFlipProb = 0.5f;
  // TransformDataset will apply each transform in a vector to the respective
  // Tensor. Thus, we need to `compose` all of the transforms so are each
  // applied only to the image
  ImageTransform trainTransforms = compose(
      {// randomly resize shortest side of image between 256 to 480 for
       // scale invariance
       fl::pkg::vision::randomResizeTransform(randomResizeMin, randomResizeMax),
       fl::pkg::vision::randomCropTransform(randomCropSize, randomCropSize),
       fl::pkg::vision::normalizeImage(mean, std),
       // Randomly flip image with probability of 0.5
       fl::pkg::vision::randomHorizontalFlipTransform(horizontalFlipProb)});
  ImageTransform valTransforms =
      compose({// Resize shortest side to 256, then take a center crop
               fl::pkg::vision::resizeTransform(randomResizeMin),
               fl::pkg::vision::centerCropTransform(randomCropSize),
               fl::pkg::vision::normalizeImage(mean, std)});

  const int64_t batchSizePerGpu = FLAGS_data_batch_size;
  const int64_t prefetchThreads = 10;
  const int64_t prefetchSize = FLAGS_data_batch_size;
  auto labelMap = getImagenetLabels(labelPath);
  auto trainDataset = fl::pkg::vision::DistributedDataset(
      imagenetDataset(trainList, labelMap, {trainTransforms}),
      worldRank,
      worldSize,
      batchSizePerGpu,
      1, // train_n_repeatedaug
      prefetchThreads,
      prefetchSize,
      fl::BatchDatasetPolicy::INCLUDE_LAST);

  auto valDataset = fl::pkg::vision::DistributedDataset(
      imagenetDataset(valList, labelMap, {valTransforms}),
      worldRank,
      worldSize,
      batchSizePerGpu,
      1, // train_n_repeatedaug
      prefetchThreads,
      prefetchSize,
      fl::BatchDatasetPolicy::INCLUDE_LAST);

  //////////////////////////
  //  Load model and optimizer
  /////////////////////////
  auto model = std::make_shared<fl::pkg::vision::Resnet50Backbone>();
  // synchronize parameters of the model so that the parameters in each process
  // is the same
  fl::allReduceParameters(model);

  // Add a hook to synchronize gradients of model parameters as they are
  // computed
  fl::distributeModuleGrads(model, reducer);

  SGDOptimizer opt(
      model->params(), FLAGS_train_lr, FLAGS_train_momentum, FLAGS_train_wd);

  auto lrScheduler = [&opt](int epoch) {
    // Adjust learning rate every 30 epoch after 30
    if (epoch == 60 || epoch == 90 || epoch == 120) {
      const float newLr = opt.getLr() * 0.1;
      LOG(INFO) << "Setting learning rate to: " << newLr;
      opt.setLr(newLr);
    }
  };

  // Small utility functions to load and save models
  auto saveModel = [&model, &isMaster](int epoch) {
    if (isMaster) {
      std::string modelPath = FLAGS_exp_checkpoint_path + std::to_string(epoch);
      LOG(INFO) << "Saving model to file: " << modelPath;
      fl::save(modelPath, model);
    }
  };

  auto loadModel = [&model](int epoch) {
    std::string modelPath = FLAGS_exp_checkpoint_path + std::to_string(epoch);
    LOG(INFO) << "Loading model from file: " << modelPath;
    fl::load(modelPath, model);
  };
  if (FLAGS_exp_checkpoint_epoch >= 0) {
    loadModel(FLAGS_exp_checkpoint_epoch);
  }

  // The main training loop
  TimeMeter timeMeter;
  TopKMeter top5Acc(5);
  TopKMeter top1Acc(1);
  AverageValueMeter trainLossMeter;
  for (int epoch = (FLAGS_exp_checkpoint_epoch + 1); epoch < FLAGS_train_epochs;
       epoch++) {
    trainDataset.resample();
    lrScheduler(epoch);

    // Get an iterator over the data
    timeMeter.resume();
    int idx = 0;
    for (auto& example : trainDataset) {
      opt.zeroGrad();
      // Make a Variable from the input Tensor.
      auto inputs = noGrad(example[kImagenetInputIdx]);

      // Get the activations from the model.
      auto output = model->forward({inputs})[0];

      // Make a Variable from the target Tensor.
      auto target = noGrad(example[kImagenetTargetIdx]);

      // Compute and record the loss.
      auto loss = categoricalCrossEntropy(output, target);

      trainLossMeter.add(loss.tensor());
      top5Acc.add(output.tensor(), target.tensor());
      top1Acc.add(output.tensor(), target.tensor());

      // Backprop, update the weights and then zero the gradients.
      loss.backward();

      if (FLAGS_distributed_enable) {
        reducer->finalize();
      }
      opt.step();

      // Compute and record the prediction error.
      double trainLoss = trainLossMeter.value()[0];
      if (++idx % 50 == 0) {
        fl::pkg::runtime::syncMeter(trainLossMeter);
        fl::pkg::runtime::syncMeter(timeMeter);
        fl::pkg::runtime::syncMeter(top5Acc);
        fl::pkg::runtime::syncMeter(top1Acc);
        double time = timeMeter.value();
        double samplePerSecond = (idx * FLAGS_data_batch_size) / time;
        FL_LOG_MASTER(INFO)
            << "Epoch " << epoch << std::setprecision(5) << " Batch: " << idx
            << " Samples per second " << samplePerSecond
            << ": Avg Train Loss: " << trainLoss
            << ": Train Top5 Accuracy( %): " << top5Acc.value()
            << ": Train Top1 Accuracy( %): " << top1Acc.value();
        top5Acc.reset();
        top1Acc.reset();
        trainLossMeter.reset();
      }
    }
    timeMeter.reset();
    timeMeter.stop();

    double valLoss, valTop1Error, valTop5Err;
    std::tie(valLoss, valTop5Err, valTop1Error) = evalLoop(model, valDataset);

    FL_LOG_MASTER(INFO) << "Epoch " << epoch << std::setprecision(5)
                        << " Validation Loss: " << valLoss
                        << " Validation Top5 Error (%): " << valTop5Err
                        << " Validation Top1 Error (%): " << valTop1Error;
    saveModel(epoch);
  }
  FL_LOG_MASTER(INFO) << "Training complete";
}

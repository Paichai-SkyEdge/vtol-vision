#include "vtol_vision/yolo_detector.hpp"

#include <gtest/gtest.h>

#include <cstdio>
#include <fstream>
#include <string>
#include <unordered_map>

namespace
{

std::string WriteTempFile(const std::string & name, const std::string & content)
{
  const std::string path = "/tmp/" + name;
  std::ofstream out(path);
  out << content;
  out.close();
  return path;
}

}  // namespace

TEST(YoloClassMap, LoadsClassNamesSequence)
{
  const std::string file = WriteTempFile(
    "class_map_seq.yaml",
    "%YAML:1.0\n---\nclass_names:\n  - basket\n  - red_cross\n");

  std::unordered_map<int, std::string> class_map;
  std::string error_message;
  const bool ok = vtol_vision::YoloDetector::LoadClassMap(file, class_map, error_message);
  EXPECT_TRUE(ok) << error_message;
  ASSERT_EQ(class_map.size(), 2U);
  EXPECT_EQ(class_map[0], "basket");
  EXPECT_EQ(class_map[1], "red_cross");
  std::remove(file.c_str());
}

TEST(YoloClassMap, LoadsClassesListWithExplicitIds)
{
  const std::string file = WriteTempFile(
    "class_map_explicit.yaml",
    "%YAML:1.0\n---\nclasses:\n  - { id: 3, name: basket }\n  - { id: 7, name: cross }\n");

  std::unordered_map<int, std::string> class_map;
  std::string error_message;
  const bool ok = vtol_vision::YoloDetector::LoadClassMap(file, class_map, error_message);
  EXPECT_TRUE(ok) << error_message;
  ASSERT_EQ(class_map.size(), 2U);
  EXPECT_EQ(class_map[3], "basket");
  EXPECT_EQ(class_map[7], "cross");
  std::remove(file.c_str());
}


#include "vtol_vision/yolo_detector.hpp"

#ifdef HAS_REALSENSE
#include <librealsense2/rs.hpp>
#endif
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/videoio.hpp>

#include <algorithm>
#include <cmath>
#include <deque>
#include <filesystem>
#include <iostream>
#include <numeric>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------
static const std::vector<std::string> CLASS_NAMES = {"basket", "mannequin"};
static const std::vector<cv::Scalar> CLASS_COLORS = {
  cv::Scalar(255, 200, 0),   // basket: cyan (BGR)
  cv::Scalar(80, 255, 0),     // mannequin: green (BGR)
};

struct Args {
  std::string model_path = "weights/yolo11n_shadow_v1_best.onnx";
  float conf = 0.25f;
  float basket_conf = 0.15f;
  float nms_threshold = 0.45f;
  int imgsz = 640;
  int width = 848;
  int height = 480;
  int depth_width = 848;
  int depth_height = 480;
  int fps = 30;
  bool no_distance = false;
  bool color_only = false;
  bool sim_mode = false;
  int stable_frames = 2;
  float track_iou = 0.25f;
  float min_area = 0.00008f;
  float max_area = 0.92f;
  bool tiled = false;
  float tile_overlap = 0.15f;
  int tile_cols = 3;
  int tile_rows = 2;
};

struct Detection {
  cv::Rect2f xyxy;
  int cls;
  float conf;
};

struct Track {
  cv::Rect2f xyxy;
  int cls;
  float conf;
  int hits = 1;
  int streak = 1;
  int missed = 0;
};

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

static float box_iou(const cv::Rect2f & a, const cv::Rect2f & b)
{
  float ix1 = std::max(a.x, b.x);
  float iy1 = std::max(a.y, b.y);
  float ix2 = std::min(a.x + a.width, b.x + b.width);
  float iy2 = std::min(a.y + a.height, b.y + b.height);
  float iw = std::max(0.0f, ix2 - ix1);
  float ih = std::max(0.0f, iy2 - iy1);
  float inter = iw * ih;
  float uni = a.width * a.height + b.width * b.height - inter;
  return uni > 0.0f ? inter / uni : 0.0f;
}

static float candidate_threshold(int cls, float dist, const Args & args)
{
  float base = (cls == 0) ? args.basket_conf : args.conf;
  if (dist > 3.0f) return std::max(0.08f, base - 0.07f);
  if (dist > 0.0f && dist < 0.8f) return base + 0.08f;
  return base;
}

// ---------------------------------------------------------------------------
// Tracking (EMA smoothing only — no motion compensation)
// ---------------------------------------------------------------------------

static std::vector<Track> update_tracks(
  std::vector<Track> & tracks, const std::vector<Detection> & dets, const Args & args)
{
  for (auto & t : tracks) t.missed++;

  std::unordered_set<int> used;
  auto sorted = dets;
  std::sort(sorted.begin(), sorted.end(),
            [](const Detection & a, const Detection & b) { return a.conf > b.conf; });

  for (const auto & det : sorted) {
    int best_idx = -1;
    float best_iou = 0.0f;
    for (int idx = 0; idx < static_cast<int>(tracks.size()); ++idx) {
      if (used.count(idx) || tracks[idx].cls != det.cls) continue;
      float iou = box_iou(tracks[idx].xyxy, det.xyxy);
      if (iou > best_iou) { best_idx = idx; best_iou = iou; }
    }
    if (best_idx >= 0 && best_iou >= args.track_iou) {
      auto & t = tracks[best_idx];
      float a = 0.55f;
      t.xyxy.x = a * t.xyxy.x + (1.0f - a) * det.xyxy.x;
      t.xyxy.y = a * t.xyxy.y + (1.0f - a) * det.xyxy.y;
      t.xyxy.width = a * t.xyxy.width + (1.0f - a) * det.xyxy.width;
      t.xyxy.height = a * t.xyxy.height + (1.0f - a) * det.xyxy.height;
      t.conf = std::max(det.conf, 0.7f * t.conf + 0.3f * det.conf);
      t.hits = std::min(t.hits + 1, 30);
      t.streak++;
      t.missed = 0;
      used.insert(best_idx);
    } else {
      tracks.push_back({det.xyxy, det.cls, det.conf});
    }
  }
  std::vector<Track> alive;
  for (const auto & t : tracks) {
    if (t.missed <= 4) alive.push_back(t);
  }
  return alive;
}

static std::vector<Track> visible_tracks(const std::vector<Track> & tracks, int stable_frames)
{
  std::vector<Track> out;
  for (const auto & t : tracks) {
    if (t.streak >= stable_frames && t.missed == 0) out.push_back(t);
  }
  return out;
}

// ---------------------------------------------------------------------------
// Tiled inference
// ---------------------------------------------------------------------------

static float center_dist(const cv::Rect2f & a, const cv::Rect2f & b)
{
  float acx = a.x + a.width * 0.5f, acy = a.y + a.height * 0.5f;
  float bcx = b.x + b.width * 0.5f, bcy = b.y + b.height * 0.5f;
  return std::hypot(acx - bcx, acy - bcy);
}

static bool has_multiscale_support(const Detection & tile_det,
                                    const std::vector<Detection> & global_dets)
{
  float tdiag = std::max(tile_det.xyxy.width, tile_det.xyxy.height);
  for (const auto & g : global_dets) {
    if (g.cls != tile_det.cls) continue;
    if (center_dist(tile_det.xyxy, g.xyxy) <= tdiag * 0.6f) return true;
  }
  return false;
}

static std::vector<cv::Rect> make_tiles(const cv::Size & frame, float overlap, int cols, int rows)
{
  int tw = frame.width / cols;
  int th = frame.height / rows;
  int ow = static_cast<int>(tw * overlap * 0.5f);
  int oh = static_cast<int>(th * overlap * 0.5f);
  std::vector<cv::Rect> tiles;
  for (int r = 0; r < rows; r++) {
    for (int c = 0; c < cols; c++) {
      int x1 = std::max(0, c * tw - ow);
      int y1 = std::max(0, r * th - oh);
      int x2 = std::min(frame.width, (c + 1) * tw + ow);
      int y2 = std::min(frame.height, (r + 1) * th + oh);
      tiles.push_back(cv::Rect(x1, y1, x2 - x1, y2 - y1));
    }
  }
  return tiles;
}

// ---------------------------------------------------------------------------
// Drawing
// ---------------------------------------------------------------------------

static void draw_detections(cv::Mat & frame, const std::vector<Track> & tracks, bool show_dist)
{
  for (const auto & t : tracks) {
    auto c = t.cls < static_cast<int>(CLASS_COLORS.size()) ? CLASS_COLORS[t.cls] : cv::Scalar(255, 255, 255);
    cv::Rect r(static_cast<int>(t.xyxy.x), static_cast<int>(t.xyxy.y),
               static_cast<int>(t.xyxy.width), static_cast<int>(t.xyxy.height));
    cv::rectangle(frame, r, c, 2);
    std::string label = CLASS_NAMES[t.cls] + " " + std::to_string(t.conf).substr(0, 4);
    if (show_dist) label += " " + std::to_string(t.conf).substr(0, 4);
    int baseline;
    cv::Size ts = cv::getTextSize(label, cv::FONT_HERSHEY_SIMPLEX, 0.55, 1, &baseline);
    cv::rectangle(frame, cv::Point(r.x, r.y - ts.height - 6),
                  cv::Point(r.x + ts.width + 4, r.y), c, -1);
    cv::putText(frame, label, cv::Point(r.x + 2, r.y - 4),
                cv::FONT_HERSHEY_SIMPLEX, 0.55, cv::Scalar(0, 0, 0), 1, cv::LINE_AA);
  }
}

static void draw_hud(cv::Mat & frame, const std::deque<float> & fps_buf, int num_det,
                     bool show_depth, bool tiled, int tile_cols, int tile_rows,
                     const std::string & mode_label)
{
  float fps = fps_buf.empty() ? 0.0f :
    std::accumulate(fps_buf.begin(), fps_buf.end(), 0.0f) / fps_buf.size();
  auto put = [&](int y, const std::string & txt, cv::Scalar clr = cv::Scalar(255, 255, 255),
                 double s = 0.7, int thk = 2) {
    cv::putText(frame, txt, cv::Point(10, y), cv::FONT_HERSHEY_SIMPLEX, s, clr, thk, cv::LINE_AA);
  };
  put(26, "FPS: " + std::to_string(fps).substr(0, 4));
  put(52, "Det: " + std::to_string(num_det));
  put(78, "Mode: " + mode_label, cv::Scalar(255, 220, 120), 0.55, 1);
  if (show_depth) put(100, "[d] depth ON", cv::Scalar(180, 180, 180), 0.55, 1);
  put(122, "[t] tiled " + std::string(tiled ? "ON" : "OFF") + " (" +
           std::to_string(tile_cols) + "x" + std::to_string(tile_rows) + ")",
      cv::Scalar(180, 180, 180), 0.55, 1);
  put(144, "[q/ESC] quit", cv::Scalar(180, 180, 180), 0.55, 1);
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

static void print_help()
{
  std::cout << "Usage: realsense_live_detect [OPTIONS]\n"
            << "  --model PATH     Model file (.onnx or .engine)\n"
            << "  --sim            Sim mode (OpenCV camera, no RealSense)\n"
            << "  --camera N       Camera index for sim mode (default 0)\n"
            << "  --video PATH     Video file input (sim mode)\n"
            << "  --conf F          Confidence (default 0.25)\n"
            << "  --basket-conf F   Basket confidence (default 0.15)\n"
            << "  --tiled           Enable tiled inference\n"
            << "  --tile-cols N     Tile columns (default 3)\n"
            << "  --tile-rows N     Tile rows (default 2)\n"
            << "  --width W         Color width (default 848)\n"
            << "  --height H        Color height (default 480)\n";
}

int main(int argc, char ** argv)
{
  Args args;
  int camera_idx = 0;
  std::string video_path;

  for (int i = 1; i < argc; i++) {
    std::string a = argv[i];
    if (a == "--model" && i + 1 < argc) args.model_path = argv[++i];
    else if (a == "--conf" && i + 1 < argc) args.conf = std::stof(argv[++i]);
    else if (a == "--basket-conf" && i + 1 < argc) args.basket_conf = std::stof(argv[++i]);
    else if (a == "--imgsz" && i + 1 < argc) args.imgsz = std::stoi(argv[++i]);
    else if (a == "--width" && i + 1 < argc) args.width = std::stoi(argv[++i]);
    else if (a == "--height" && i + 1 < argc) args.height = std::stoi(argv[++i]);
    else if (a == "--depth-width" && i + 1 < argc) args.depth_width = std::stoi(argv[++i]);
    else if (a == "--depth-height" && i + 1 < argc) args.depth_height = std::stoi(argv[++i]);
    else if (a == "--fps" && i + 1 < argc) args.fps = std::stoi(argv[++i]);
    else if (a == "--no-distance") args.no_distance = true;
    else if (a == "--color-only") args.color_only = true;
    else if (a == "--sim") args.sim_mode = true;
    else if (a == "--camera" && i + 1 < argc) camera_idx = std::stoi(argv[++i]);
    else if (a == "--video" && i + 1 < argc) video_path = argv[++i];
    else if (a == "--stable-frames" && i + 1 < argc) args.stable_frames = std::stoi(argv[++i]);
    else if (a == "--track-iou" && i + 1 < argc) args.track_iou = std::stof(argv[++i]);
    else if (a == "--min-area" && i + 1 < argc) args.min_area = std::stof(argv[++i]);
    else if (a == "--max-area" && i + 1 < argc) args.max_area = std::stof(argv[++i]);
    else if (a == "--tiled") args.tiled = true;
    else if (a == "--tile-overlap" && i + 1 < argc) args.tile_overlap = std::stof(argv[++i]);
    else if (a == "--tile-cols" && i + 1 < argc) args.tile_cols = std::stoi(argv[++i]);
    else if (a == "--tile-rows" && i + 1 < argc) args.tile_rows = std::stoi(argv[++i]);
    else if (a == "-h" || a == "--help") { print_help(); return 0; }
  }

  // --- Load YOLO ---
  vtol_vision::YoloDetector detector;
  std::unordered_map<int, std::string> class_map = {{0, "basket"}, {1, "mannequin"}};
  for (auto & p : {"config/basket_mannequin_class_map.yaml",
                    "../config/basket_mannequin_class_map.yaml",
                    "../../config/basket_mannequin_class_map.yaml"}) {
    std::string err;
    if (std::filesystem::exists(p)) {
      vtol_vision::YoloDetector::LoadClassMap(p, class_map, err);
      break;
    }
  }

  if (!detector.Initialize(args.model_path, class_map, args.imgsz, args.conf, args.nms_threshold)) {
    std::cerr << "Failed to init YOLO. Ensure model exists: " << args.model_path << std::endl;
    return 1;
  }

  std::string mode_label = (detector.backend() == vtol_vision::YoloDetector::Backend::TensorRT)
    ? "TRT" : "DNN";

  // --- Open Camera ---
  cv::VideoCapture cap;
  bool has_depth = false;

#ifdef HAS_REALSENSE
  rs2::pipeline pipe;
  rs2::config cfg;
  rs2::align * align = nullptr;

  if (!args.sim_mode) {
    cfg.enable_stream(RS2_STREAM_COLOR, args.width, args.height, RS2_FORMAT_BGR8, args.fps);
    if (!args.color_only) {
      cfg.enable_stream(RS2_STREAM_DEPTH, args.depth_width, args.depth_height, RS2_FORMAT_Z16, args.fps);
      has_depth = true;
    }
    try {
      pipe.start(cfg);
      align = new rs2::align(RS2_STREAM_COLOR);
      mode_label += " +RS";
      std::cout << "Camera: RealSense D435i " << args.width << "x" << args.height << "@" << args.fps << std::endl;
    } catch (const rs2::error & e) {
      std::cerr << "RealSense failed: " << e.what() << " — falling back to sim mode" << std::endl;
      args.sim_mode = true;
    }
  }
#endif

  if (args.sim_mode) {
    if (!video_path.empty()) {
      cap.open(video_path);
      mode_label += " +Video";
    } else {
      cap.open(camera_idx);
      mode_label += " +Cam";
    }
    if (!cap.isOpened()) {
      std::cerr << "Failed to open camera " << camera_idx << std::endl;
      return 1;
    }
    cap.set(cv::CAP_PROP_FRAME_WIDTH, args.width);
    cap.set(cv::CAP_PROP_FRAME_HEIGHT, args.height);
    cap.set(cv::CAP_PROP_FPS, args.fps);
    std::cout << "Camera: sim index=" << camera_idx << " " << args.width << "x" << args.height << std::endl;
  }

  // --- Main loop ---
  std::deque<float> fps_buf;
  std::vector<Track> tracks;
  bool show_depth = false;
  bool tiled_enabled = args.tiled;
  double t_prev = static_cast<double>(cv::getTickCount()) / cv::getTickFrequency();

  const std::string win = "basket/mannequin detector [C++]";
  std::cout << "Streaming — q/ESC quit, d=depth, t=tiled" << std::endl;

  while (true) {
    cv::Mat color, depth_mat;

#ifdef HAS_REALSENSE
    if (!args.sim_mode) {
      rs2::frameset frames;
      try {
        frames = pipe.wait_for_frames(5000);
      } catch (...) { break; }
      auto aligned = align ? align->process(frames) : frames;
      auto cf = aligned.get_color_frame();
      if (!cf) continue;
      color = cv::Mat(cf.get_height(), cf.get_width(), CV_8UC3,
                      const_cast<void *>(cf.get_data()), cf.get_stride_in_bytes()).clone();
      if (has_depth) {
        auto df = aligned.get_depth_frame();
        if (df) {
          depth_mat = cv::Mat(df.get_height(), df.get_width(), CV_16UC1,
                              const_cast<void *>(df.get_data()), df.get_stride_in_bytes()).clone();
        }
      }
    }
#endif

    if (args.sim_mode) {
      cap >> color;
      if (color.empty()) {
        std::cerr << "Camera lost, retrying..." << std::endl;
        cap.release();
        if (!video_path.empty()) cap.open(video_path);
        else cap.open(camera_idx);
        if (!cap.isOpened()) break;
        continue;
      }
    }

    if (color.empty()) continue;
    cv::Mat disp = color.clone();

    // --- Inference ---
    std::vector<Detection> detections;
    std::vector<cv::Rect> tiles;

    if (tiled_enabled) {
      tiles = make_tiles(color.size(), args.tile_overlap, args.tile_cols, args.tile_rows);

      // Global pass → detections + validation candidates
      std::vector<Detection> global_candidates;
      for (const auto & c : detector.Infer(color)) {
        Detection d{cv::Rect2f(static_cast<float>(c.bbox.x), static_cast<float>(c.bbox.y),
                                static_cast<float>(c.bbox.width), static_cast<float>(c.bbox.height)),
                    c.class_id, c.score};
        global_candidates.push_back(d);
        if (d.conf >= candidate_threshold(d.cls, 0.0f, args)) detections.push_back(d);
      }

      // Tile passes — multiscale-validated only
      for (const auto & t : tiles) {
        if (!depth_mat.empty()) {
          cv::Mat td = depth_mat(t);
          if (cv::countNonZero(td > 0) < td.total() * 0.05f) continue;
        }
        auto cands = detector.Infer(color(t));
        for (const auto & c : cands) {
          Detection d{cv::Rect2f(static_cast<float>(t.x + c.bbox.x),
                                  static_cast<float>(t.y + c.bbox.y),
                                  static_cast<float>(c.bbox.width),
                                  static_cast<float>(c.bbox.height)),
                      c.class_id, c.score};
          if (has_multiscale_support(d, global_candidates)) detections.push_back(d);
        }
      }
    } else {
      for (const auto & c : detector.Infer(color)) {
        detections.push_back({cv::Rect2f(static_cast<float>(c.bbox.x), static_cast<float>(c.bbox.y),
                                          static_cast<float>(c.bbox.width), static_cast<float>(c.bbox.height)),
                              c.class_id, c.score});
      }
    }

    // Filter: area ratio + confidence threshold
    float fa = static_cast<float>(color.cols * color.rows);
    std::vector<Detection> filtered;
    for (auto & d : detections) {
      if (d.xyxy.width * d.xyxy.height / fa < args.min_area) continue;
      if (d.xyxy.width * d.xyxy.height / fa > args.max_area) continue;
      if (d.conf < candidate_threshold(d.cls, 0.0f, args)) continue;
      filtered.push_back(d);
    }

    // Per-class best (max 1 each)
    std::unordered_map<int, Detection> best_pc;
    for (const auto & d : filtered) {
      auto it = best_pc.find(d.cls);
      if (it == best_pc.end() || d.conf > it->second.conf) best_pc[d.cls] = d;
    }
    detections.clear();
    for (auto & kv : best_pc) detections.push_back(kv.second);

    tracks = update_tracks(tracks, detections, args);
    int rs = tiled_enabled ? std::max(args.stable_frames, 2) : args.stable_frames;
    auto shown = visible_tracks(tracks, rs);

    // Draw
    if (tiled_enabled) {
      for (const auto & t : tiles) cv::rectangle(disp, t, cv::Scalar(40, 160, 255), 1);
    }
    draw_detections(disp, shown, !args.no_distance);

    double t_now = static_cast<double>(cv::getTickCount()) / cv::getTickFrequency();
    fps_buf.push_back(1.0f / std::max(static_cast<float>(t_now - t_prev), 1e-6f));
    t_prev = t_now;
    if (fps_buf.size() > 30) fps_buf.pop_front();

    draw_hud(disp, fps_buf, static_cast<int>(shown.size()),
             show_depth, tiled_enabled, args.tile_cols, args.tile_rows, mode_label);

    if (show_depth && !depth_mat.empty()) {
      cv::Mat d8u, dcolor;
      depth_mat.convertTo(d8u, CV_8U, 0.03);
      cv::applyColorMap(d8u, dcolor, cv::COLORMAP_JET);
      int sw = args.width / 3, sh = args.height / 3;
      cv::resize(dcolor, dcolor, cv::Size(sw, sh));
      dcolor.copyTo(disp(cv::Rect(args.width - sw, args.height - sh, sw, sh)));
    }

    cv::imshow(win, disp);
    int key = cv::waitKey(1) & 0xFF;
    if (key == 'q' || key == 27) break;
    if (key == 'd') show_depth = !show_depth;
    if (key == 't') { tiled_enabled = !tiled_enabled; tracks.clear(); fps_buf.clear(); }
  }

#ifdef HAS_REALSENSE
  if (!args.sim_mode) { pipe.stop(); delete align; }
#endif
  cv::destroyAllWindows();
  return 0;
}

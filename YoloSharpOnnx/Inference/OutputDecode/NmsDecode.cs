using OpenCvSharp;
using OpenCvSharp.Dnn;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Reflection.Emit;
using System.Text;
using System.Threading.Tasks;
using YoloSharpOnnx.DataResult;
using YoloSharpOnnx.Inference.Detect.Models;
using YoloSharpOnnx.Models;
using static System.Formats.Asn1.AsnWriter;

namespace YoloSharpOnnx.Inference.OutputDecode
{
    internal class NmsDecode
    {
        private readonly int _numAnchors;
        private readonly int _numAnchors2;
        private readonly int _numAnchors3;
        private readonly int _numAnchors4;
        private readonly int _classCount;

        private readonly YoloConfig _yoloConfig;

        public NmsDecode(OnnxModel onnx, YoloConfig yoloConfig)
        {
            _numAnchors = (int)onnx.OutputShape0[2];

            _numAnchors2 = _numAnchors * 2;
            _numAnchors3 = _numAnchors * 3;
            _numAnchors4 = _numAnchors * 4;

            _classCount = onnx.Labels.Length;
            _yoloConfig = yoloConfig;
        }


        public int[] Decode(ReadOnlySpan<float> output0, PreDetectResult preResult, List<Rect> boxes, List<float> scores, List<int> classIds, List<int> ids = null)
        {
            for (int i = 0; i < _numAnchors; i++)
            {
                int classOffset = i + _numAnchors4;
                float maxScore = 0f;
                int classId = -1;

                for (var c = 0; c < _classCount; c++, classOffset += _numAnchors)
                {
                    float score = output0[classOffset];

                    if (score > maxScore)
                    {
                        maxScore = score;
                        classId = c;
                    }
                }
                if (maxScore < _yoloConfig.Confidence) continue;

                float cx = output0[i] - preResult.PadX;
                float cy = output0[_numAnchors + i] - preResult.PadY;
                float w = output0[_numAnchors2 + i];
                float h = output0[_numAnchors3 + i];

                int x = (int)((cx - w / 2f) / preResult.Scale);
                int y = (int)((cy - h / 2f) / preResult.Scale);
                int width = (int)(w / preResult.Scale);
                int height = (int)(h / preResult.Scale);

                // Ensure coordinates are within image bounds
                x = Math.Max(0, x);
                y = Math.Max(0, y);
                width = Math.Min(width, preResult.ImageWidth - x);
                height = Math.Min(height, preResult.ImageHeight - y);

                // Add the class ID, score, and box coordinates to the respective lists
                if (width > 0 && height > 0)
                {
                    classIds.Add(classId);
                    scores.Add(maxScore);
                    boxes.Add(new Rect(x, y, width, height));

                    ids?.Add(i);
                }
            }

            // 非极大值抑制
            int[] indices = [];
            if (boxes.Count > 0)
            {
                CvDnn.NMSBoxes(boxes, scores, _yoloConfig.Confidence, _yoloConfig.IoU, out indices);
            }
            return indices;
        }


        public List<ObbResult> Decode(ReadOnlySpan<float> output0, PreDetectResult preResult, List<ObbResult> obbResults, LabelModel[] labels)
        {
            int angleIndex = (4 + _classCount) * _numAnchors;//cx, cy, w, h

            for (int i = 0; i < _numAnchors; i++)
            {
                int classOffset = i + _numAnchors4;
                float maxScore = 0f;
                int classId = -1;

                for (var c = 0; c < _classCount; c++, classOffset += _numAnchors)
                {
                    float score = output0[classOffset];

                    if (score > maxScore)
                    {
                        maxScore = score;
                        classId = c;
                    }
                }
                if (maxScore < _yoloConfig.Confidence) continue;

                float cx = (output0[i] - preResult.PadX) / preResult.Scale;
                float cy = (output0[_numAnchors + i] - preResult.PadY) / preResult.Scale;
                float w = output0[_numAnchors2 + i] / preResult.Scale;
                float h = output0[_numAnchors3 + i] / preResult.Scale;

                float angle = output0[angleIndex + i]; // 弧度

                obbResults.Add(new ObbResult
                {
                    ClassId = classId,
                    ClassName = labels[classId].Name,
                    Confidence = maxScore,
                    Center = new Point2f(cx, cy),
                    Width = w,
                    Height = h,
                    Angle = YoloUtils.ToDegree(angle)
                });
            }

            return ApplyRotatedNms(obbResults);
        }

        private List<ObbResult> ApplyRotatedNms(List<ObbResult> candidates)
        {
            // 按照得分降序排列
            candidates.Sort((x, y) => y.Confidence.CompareTo(x.Confidence));

            var results = new List<ObbResult>();
            bool[] skipped = new bool[candidates.Count];

            for (int i = 0; i < candidates.Count; i++)
            {
                if (skipped[i]) continue;

                var c1 = candidates[i];
                results.Add(c1);

                // OpenCV 的 RotatedRect 角度参数需要角度制 (Degrees)

                RotatedRect rect1 = ToCvRect(c1);
                double area1 = (double)c1.Width * c1.Height;

                for (int j = i + 1; j < candidates.Count; j++)
                {
                    if (skipped[j]) continue;

                    var c2 = candidates[j];

                    if (c1.ClassId != c2.ClassId) continue;

                    RotatedRect rect2 = ToCvRect(c2);
                    double area2 = (double)c2.Width * c2.Height;

                    // 计算两个旋转矩形的交集顶点
                    var intersectType = Cv2.RotatedRectangleIntersection(rect1, rect2, out Point2f[] intersectPoints);

                    if (intersectType == RectanglesIntersectTypes.None || intersectPoints == null || intersectPoints.Length < 3)
                    {
                        continue;
                    }

                    double intersectArea = Math.Abs(Cv2.ContourArea(intersectPoints));

                    // 计算旋转 IoU
                    double iou = intersectArea / (area1 + area2 - intersectArea);

                    if (iou > _yoloConfig.IoU)
                    {
                        skipped[j] = true; // 剔除高重叠的低分框
                    }
                }
            }

            return results;
        }

        private static RotatedRect ToCvRect(ObbResult box)
        {
            return new RotatedRect(box.Center, new Size2f(box.Width, box.Height), box.Angle);
        }

    }
}

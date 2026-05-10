using System;
using System.Collections;
using System.Collections.Generic;
using System.Collections.Specialized;
using System.Text;
using static OpenCvSharp.FileStorage;

namespace YoloSharpOnnx.DataResult
{
    public static class ResultExtensions
    {
        public static string Summary(this List<DetectionResult> boxes)
        {
            if (boxes == null || boxes.Count == 0)
                return string.Empty;

            return DetectionToString(boxes);

        }

        public static string Summary(this List<ClsResult> clsList)
        {
            if (clsList == null || clsList.Count == 0)
                return string.Empty;
            return ClsToString(clsList);

        }

        public static string SummaryOrder(this List<DetectionResult> boxes)
        {
            if (boxes == null || boxes.Count == 0)
                return string.Empty;

            OrderList(boxes);
            return DetectionToString(boxes);
        }

        public static string SummaryOrder(this List<ClsResult> clsList)
        {
            if (clsList == null || clsList.Count == 0)
                return string.Empty;

            OrderList(clsList);
            return ClsToString(clsList);

        }

        private static void OrderList<T>(List<T> list) where T : IYoloResult
        {
            list.Sort((a, b) => a.ClassName.CompareTo(b.ClassName));
        }
        private static string DetectionToString(List<DetectionResult> boxes)
        {
            var dict = boxes.GroupBy(p => p.ClassName).Select(p => $"{p.Count()} {p.Key}").ToList();
            string confs = string.Join(", ", boxes.Select(p => Math.Round(p.Confidence, 2)));
            return $"{string.Join(", ", dict)} [{confs}]";
            
        }
        private static string ClsToString(List<ClsResult> clsList)
        {
            string confs = string.Join(", ", clsList.Select(p => $"{p.ClassName} {Math.Round(p.Confidence, 2)}"));
            return confs;

        }
    }
}

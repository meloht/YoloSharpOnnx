using System;
using System.Collections;
using System.Collections.Generic;
using System.Text;

namespace YoloSharpOnnx.DataResult
{
    public static class ResultExtensions
    {
        public static string Summary(this ICollection<DetectionResult> boxes)
        {
            if (boxes == null || boxes.Count == 0)
                return string.Empty;
     
            var dict = boxes.GroupBy(p => p.ClassName).Select(p => $"{p.Count()} {p.Key}").ToList();
            string confs = string.Join(", ", boxes.Select(p => Math.Round(p.Confidence, 2)));
            return $"{string.Join(", ", dict)} [{confs}]";

        }

        public static string Summary(this ICollection<ClsResult> clsList)
        {
            if (clsList == null || clsList.Count == 0)
                return string.Empty;

            var dict = clsList.GroupBy(p => p.ClassName).Select(p => $"{p.Count()} {p.Key}").ToList();
            string confs = string.Join(", ", clsList.Select(p => Math.Round(p.Confidence, 2)));
            return $"{string.Join(", ", dict)} [{confs}]";

        }
    }
}

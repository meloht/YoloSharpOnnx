using System;
using System.Collections.Generic;
using System.Text;
using YoloSharpOnnx.DataResult;

namespace YoloSharpOnnx
{
    public class YoloUtils
    {
        public static string GetResult(List<DetectionResult> list)
        {
            if (list == null || list.Count == 0)
                return string.Empty;

            var dict = list.GroupBy(p => p.ClassName).Select(p => $"{p.Count()} {p.Key}").ToList();
            string confs = string.Join(", ", list.Select(p => Math.Round(p.Confidence, 2)));
            return $"{string.Join(", ", dict)} [{confs}]";
        }
    }
}

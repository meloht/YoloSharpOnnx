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



        public static List<string> GetFilesFromDirectory(string path, string[] exts)
        {
            List<string> list = new List<string>();
            HashSet<string> set = new HashSet<string>(exts);
            GetFiles(list, path, set);
            return list;

        }

        public static void GetFiles(List<string> list, string path, HashSet<string> extSet)
        {
            DirectoryInfo directory = new DirectoryInfo(path);
            var files = directory.GetFiles();

            foreach (var item in files)
            {
                string fileExt = item.Extension.ToLower();
                if (extSet.Contains(fileExt))
                {
                    list.Add(item.FullName);
                }
            }
            var subDirectories = Directory.GetDirectories(path);

            foreach (string subDir in subDirectories)
            {
                GetFiles(list, subDir, extSet);
            }
        }
    }
}

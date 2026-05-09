using System;
using System.Collections.Generic;
using System.Text;
using YoloSharpOnnx.DataResult;
using static System.Net.WebRequestMethods;

namespace YoloSharpOnnx
{
    public class YoloUtils
    {
        public static string GetDetectResult(List<DetectionResult> list)
        {
            return list.Summary();
        }
        public static string GetClsResult(List<ClsResult> list)
        {
            return list.Summary();
        }


        public static List<string> GetFilesFromDirectory(string path, string[] exts)
        {
            List<string> list = new List<string>();
            HashSet<string> set = new HashSet<string>(exts);
            GetFiles(list, path, set);
            return list;

        }

        public static List<string> GetFilesFromListPaths(List<string> images, string[] exts)
        {

            List<string> list = new List<string>();
            HashSet<string> extSet = new HashSet<string>(exts);
            foreach (var item in images)
            {
                string ext = Path.GetExtension(item);
                string fileExt = ext.ToLower();
                if (extSet.Contains(fileExt))
                {
                    list.Add(item);
                }
            }
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

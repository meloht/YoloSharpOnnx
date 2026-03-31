using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace YoloSharpOnnx
{
    internal class YoloValidation
    {
        public static void ValidationImagePath(string imagePath, YoloConfig yoloConfig)
        {
            if (string.IsNullOrWhiteSpace(imagePath))
            {
                throw new ArgumentNullException($"imagePath is null or empty");
            }
            if (!File.Exists(imagePath))
            {
                throw new DirectoryNotFoundException($"{imagePath} file is not found");
            }
            string ext = Path.GetExtension(imagePath);
            if (yoloConfig.ImageExtsBatch.AsSpan().IndexOf(ext) == -1)
            {
                throw new ArgumentNullException($"imagePath is not a image for ext({string.Join(',', yoloConfig.ImageExtsBatch)})");
            }
        }

        public static List<string> ValidationImageBatch(string imgDir, int batchSize, YoloConfig yoloConfig)
        {
            if (string.IsNullOrWhiteSpace(imgDir))
            {
                throw new ArgumentNullException($"imgDir is null or empty");
            }
            if (!Directory.Exists(imgDir))
            {
                throw new DirectoryNotFoundException($"{imgDir} directory not found");
            }
            if (batchSize <= 0)
            {
                throw new ArgumentNullException("batchSize must be greater than zero");
            }

            var files = YoloUtils.GetFilesFromDirectory(imgDir, yoloConfig.ImageExtsBatch);
            if (files.Count == 0)
            {
                throw new ArgumentNullException($"there no any images in the directory for image ext({string.Join(',', yoloConfig.ImageExtsBatch)})");
            }
            return files;
        }

        public static void ValidationImageListPath(List<string> list, YoloConfig yoloConfig)
        {
            if (list == null || list.Count == 0)
            {
                throw new ArgumentNullException($"images is invalid for image ext({string.Join(',', yoloConfig.ImageExtsBatch)})");
            }

        }
    }
}

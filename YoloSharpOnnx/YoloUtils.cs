using OpenCvSharp;
using System;
using System.Collections.Generic;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using System.Text;
using System.Threading.Channels;
using YoloSharpOnnx.DataResult;
using YoloSharpOnnx.Inference;
using YoloSharpOnnx.Inference.OutputDecode;
using YoloSharpOnnx.Models;

namespace YoloSharpOnnx
{
    public static class YoloUtils
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

        public static BoundedChannelOptions GetChannelOptions(int batchPoolSize)
        {
            var channelOptions = new BoundedChannelOptions(batchPoolSize)
            {
                SingleWriter = false,
                SingleReader = true,
                AllowSynchronousContinuations = false,
                FullMode = BoundedChannelFullMode.Wait
            };

            return channelOptions;
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        internal static float ToDegree(float angle)
        {
            return angle * 180.0f / MathF.PI;
        }
        internal static void ClearList(PostResultArray resultArray)
        {
            resultArray.Boxes.Clear();
            resultArray.Scores.Clear();
            resultArray.ClassIds.Clear();
            resultArray.Ids?.Clear();
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



        public static unsafe void MatToBytes(Mat mat, byte[] buffer)
        {
            int width = mat.Cols;
            int height = mat.Rows;
            int channels = mat.Channels();

            byte* ptr = (byte*)mat.DataPointer;
            fixed (byte* data = buffer)
            {
                int hw = width * height;

                long step = mat.Step();
                for (int y = 0; y < height; y++)
                {
                    byte* rowPtr = ptr + y * step;
                    int rowOffset = y * width;

                    for (int x = 0; x < width; x++)
                    {
                        byte* pixel = rowPtr + x * channels;
                        for (int c = 0; c < channels; c++)
                        {
                            int offset = hw * c + rowOffset + x;
                            data[offset] = pixel[c];
                        }

                    }
                }
            }

        }

        public static byte[] PackMask(byte[] src)
        {
            int packedLength = (src.Length + 7) >> 3;

            byte[] packed = new byte[packedLength];

            for (int i = 0; i < src.Length; i++)
            {
                if (src[i] != 0)
                {
                    packed[i >> 3] |= (byte)(1 << (i & 7));
                }
            }

            return packed;
        }

        public static byte[] UnpackMask(byte[] packed, int pixelCount)
        {
            byte[] dst = new byte[pixelCount];

            for (int i = 0; i < pixelCount; i++)
            {
                dst[i] =
                    (packed[i >> 3] & (1 << (i & 7))) != 0
                    ? (byte)255
                    : (byte)0;
            }

            return dst;
        }

        public static void UnpackMask(byte[] packed, byte[] dst, int pixelCount)
        {
            for (int i = 0; i < pixelCount; i++)
            {
                dst[i] =
                    (packed[i >> 3] & (1 << (i & 7))) != 0
                    ? (byte)255
                    : (byte)0;
            }
        }

    }
}

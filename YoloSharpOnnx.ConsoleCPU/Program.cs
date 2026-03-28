using OpenCvSharp;
using YoloSharpOnnx.DataResult;
using YoloSharpOnnx.Providers;

namespace YoloSharpOnnx.ConsoleCPU
{
    internal class Program
    {
        static void Main(string[] args)
        {
            Console.WriteLine("Hello, World!");
            //TestInfer();
            TestBatchInfer();
            //TestInferPerf();
            //using Mat image = Cv2.ImRead("bus.jpg");
            //using YoloSharp yolo = new YoloSharp(new ExecutionProviderCPU("yolo11n.onnx"));

            //List<DetectionResult> res = yolo.RunDetect(image);
            //yolo.DrawDetections(image, res);
            //Cv2.ImWrite("bus_res.jpg", image);
            //string printString = YoloUtils.GetResult(res);
            //Console.WriteLine(printString);
        }

        private static void TestInfer()
        {
            string modelPath = @"D:\code\model\best.onnx";
            string dir = @"D:\code\model\TestImages";

            DirectoryInfo directory = new DirectoryInfo(dir);
            var files = directory.GetFiles();

            System.Diagnostics.Stopwatch _stopwatch = new System.Diagnostics.Stopwatch();
            System.Diagnostics.Stopwatch _stopwatchTotal = new System.Diagnostics.Stopwatch();
            _stopwatchTotal.Start();
            using (YoloSharp yolo = new YoloSharp(new ExecutionProviderCPU(modelPath)))
            {
                foreach (var item in files)
                {
                    string filePath = item.Extension.ToLower();
                    if (filePath.EndsWith(".jpg") || filePath.EndsWith(".png"))
                    {
                        _stopwatch.Restart();
                        var res = yolo.RunDetect(item.FullName);
                        _stopwatch.Stop();
                        string ans = YoloUtils.GetResult(res);
                        Console.WriteLine($"{ans}, time:{_stopwatch.ElapsedMilliseconds}");
                    }
                }
            }
            _stopwatchTotal.Stop();

            Console.WriteLine($"time:{_stopwatchTotal.Elapsed}");

        }

        private static void TestInferPerf()
        {
            string modelPath = @"D:\code\model\best.onnx";
            string dir = @"D:\code\model\TestImages";

            DirectoryInfo directory = new DirectoryInfo(dir);
            var files = directory.GetFiles();


            using (YoloSharp yolo = new YoloSharp(new ExecutionProviderCPU(modelPath)))
            {
                foreach (var item in files)
                {
                    string filePath = item.Extension.ToLower();
                    if (filePath.EndsWith(".jpg") || filePath.EndsWith(".png"))
                    {

                        var res = yolo.RunDetectWithTime(item.FullName);

                        Console.WriteLine($"{res.ToString()}, {res.SpeedResult.ToString()}");
                    }
                }
            }

        }

        private static void TestBatchInfer()
        {
            string modelPath = @"D:\code\model\best.onnx";
            string dir = @"D:\code\model\TestImages";

            DirectoryInfo directory = new DirectoryInfo(dir);
            var files = directory.GetFiles();

            System.Diagnostics.Stopwatch _stopwatch = new System.Diagnostics.Stopwatch();
            _stopwatch.Start();
            using (YoloSharp yolo = new YoloSharp(new ExecutionProviderCPU(modelPath)))
            {
                yolo.BatchDetectItemCompleted += Yolo_BatchDetectCompleted;
               
                yolo.RunBatchDetect(dir, 30);

            }
            _stopwatch.Stop();

            Console.WriteLine($"time:{_stopwatch.Elapsed}");
        }

        private static void Yolo_BatchDetectCompleted(object? sender, DetectionBatchResult e)
        {
            string ans = YoloUtils.GetResult(e.Results);
            Console.WriteLine(ans);
        }
    }
}

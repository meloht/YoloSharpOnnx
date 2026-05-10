using OpenCvSharp;
using YoloSharpOnnx.DataResult;
using YoloSharpOnnx.Providers;

namespace YoloSharpOnnx.ConsoleCPU
{
    internal class Program
    {
        static string modelPath = @"D:\code\model\best.onnx";
        static string dir = @"D:\code\model\TestImages";
        static void Main(string[] args)
        {

            Console.WriteLine("Hello, World!");
            //TestInfer();
            //TestBatchInfer();
            //TestInferCls();
            //TestInferPerf();
            //string img = @"D:\code\model\COCO2017\train2017\train2017\000000253890.jpg";
            //using Mat image = Cv2.ImRead(img);
            //using YoloSharp yolo = new YoloSharp(new ExecutionProviderCPU(@"D:\code\YoloSharpOnnx\YoloSharpOnnx.TestCommon\TestData\Models\yolo11n.onnx"));

            //List<DetectionResult> res = yolo.RunDetect(image);
            //yolo.DrawDetections(image, res);

            //Cv2.ImWrite($"det_{Path.GetFileName(img)}", image);
            //string printString = res.Summary();
            //Console.WriteLine(printString);
        }

        private static void TestInfer()
        {


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
                        string ans = res.Summary();
                        Console.WriteLine($"{ans}, time:{_stopwatch.ElapsedMilliseconds}");
                    }
                }
            }
            _stopwatchTotal.Stop();

            Console.WriteLine($"time:{_stopwatchTotal.Elapsed}");

        }

        private static void TestInferCls()
        {
            string model = @"D:\code\YoloSharpOnnx\YoloSharpOnnx.TestCommon\TestData\Models\yolo26n-cls.onnx";
            string img = @"D:\code\model\COCO2017\train2017\train2017\000000202178.jpg";
            using Mat image = Cv2.ImRead(img);
            using YoloSharp yolo = new YoloSharp(new ExecutionProviderCPU(model));
            var res = yolo.RunClassifyWithTime(image);
            Console.WriteLine($"{res.ToString()}, {res.SpeedResult.ToString()}");
            yolo.DrawClassification(image, res.Items);
            Cv2.ImWrite($"cls_{Path.GetFileName(img)}", image);
        }

        private static void TestInferPerf()
        {
            DirectoryInfo directory = new DirectoryInfo(dir);
            var files = directory.GetFiles();
            System.Diagnostics.Stopwatch _stopwatch = new System.Diagnostics.Stopwatch();
            _stopwatch.Start();

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
            _stopwatch.Stop();

            Console.WriteLine($"time:{_stopwatch.Elapsed}");

        }

        private static void TestBatchInfer()
        {
            DirectoryInfo directory = new DirectoryInfo(dir);
            var files = directory.GetFiles();

            System.Diagnostics.Stopwatch _stopwatch = new System.Diagnostics.Stopwatch();
            _stopwatch.Start();
            using (YoloSharp yolo = new YoloSharp(new ExecutionProviderCPU(modelPath)))
            {
                yolo.YoloConfiguration.BatchPoolSize = 30;
               
                yolo.RunBatchDetect(dir);

            }
            _stopwatch.Stop();

            Console.WriteLine($"time:{_stopwatch.Elapsed}");
        }


    }
}

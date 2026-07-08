using YoloSharpOnnx.DataResult;
using YoloSharpOnnx.Providers;

namespace YoloSharpOnnx.ConsoleGPU
{
    internal class Program
    {
        static string modelPath = @"C:\code\model\best.onnx";
        static string dir = @"C:\code\model\TestImages_300";
        static int _deviceId = 0;
        static void Main(string[] args)
        {
            Console.WriteLine("Hello, World!");
            //TestBatchInferTensorRT();
            //TestInferPerf();
            TestInfer();
            //TestInferSeg();
            Console.WriteLine("end!");
            Console.ReadKey();
        }

        private static void TestInfer()
        {
            DirectoryInfo directory = new DirectoryInfo(dir);
            var files = directory.GetFiles();
            System.Diagnostics.Stopwatch _stopwatchTotal = new System.Diagnostics.Stopwatch();
            _stopwatchTotal.Start();

            System.Diagnostics.Stopwatch _stopwatch = new System.Diagnostics.Stopwatch();
            using (YoloSharp yolo = new YoloSharp(new ExecutionProviderCUDA(modelPath, _deviceId)))
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
        private static void TestBatchInferTensorRT()
        {

            DirectoryInfo directory = new DirectoryInfo(dir);
            var files = directory.GetFiles();

            System.Diagnostics.Stopwatch _stopwatch = new System.Diagnostics.Stopwatch();
            _stopwatch.Start();
            int num = files.Length;

            Dictionary<string, string> dict = new Dictionary<string, string>();
            dict.Add("device_id", _deviceId.ToString());
            dict.Add("trt_engine_cache_enable", "true");
            dict.Add("trt_dump_ep_context_model", "true");
            dict.Add("trt_ep_context_file_path", Path.Combine(Directory.GetCurrentDirectory(), "trt_engine_cache"));
            dict.Add("trt_engine_cache_path", Path.Combine(Directory.GetCurrentDirectory(), "trt_engine_cache"));
            dict.Add("trt_engine_cache_prefix", "YoloSharpOnnx");
            dict.Add("trt_auxiliary_streams", "0");
            dict.Add("trt_builder_optimization_level", "3");
            using (YoloSharp yolo = new YoloSharp(new ExecutionProviderTensorRT(modelPath, _deviceId, dict)))
            {
                yolo.YoloConfiguration.BatchPoolSize = 30;


                var list = yolo.RunBatchDetect(dir, receiveAction: ReceiveProcess);

            }
            _stopwatch.Stop();

            Console.WriteLine($"detect {num} images, time:{_stopwatch.Elapsed}");
        }


        private static void ReceiveProcess(DetectionBatchResult e)
        {

            long cost = DateTimeOffset.UtcNow.ToUnixTimeMilliseconds() - e.StartTimestamp;
            string ans = e.Results.Summary();
            Console.WriteLine($"{ans} time:{cost}ms");

        }
        private static void TestInferPerf()
        {
            DirectoryInfo directory = new DirectoryInfo(dir);
            var files = directory.GetFiles();
            System.Diagnostics.Stopwatch _stopwatchTotal = new System.Diagnostics.Stopwatch();
            _stopwatchTotal.Start();

            using (YoloSharp yolo = new YoloSharp(new ExecutionProviderCUDA(modelPath, _deviceId)))
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
            _stopwatchTotal.Stop();

            Console.WriteLine($"time:{_stopwatchTotal.Elapsed}");

        }
        private static void TestInferSeg()
        {
            System.Diagnostics.Stopwatch _stopwatch = new System.Diagnostics.Stopwatch();
            int num = 0;
            var files = Directory.GetFiles(@"C:\code\model\val2017");
            using (YoloSharp yolo = new YoloSharp(new ExecutionProviderCUDA(@"C:\code\model\yolo11n-seg.onnx", _deviceId)))
            {

                _stopwatch.Start();
                for (int i = 0; i < files.Length; i++)
                {

                    var res = yolo.RunSegmentWithTime(files[i]);
                    Console.WriteLine($"{i + 1} {res.SpeedResult}");
                }


                num = files.Length;
                _stopwatch.Stop();

            }


            Console.WriteLine($"detect {num} images, time:{_stopwatch.Elapsed}");
        }

        private static void TestInferPerfTensorRT()
        {
            DirectoryInfo directory = new DirectoryInfo(dir);
            var files = directory.GetFiles();
            System.Diagnostics.Stopwatch _stopwatchTotal = new System.Diagnostics.Stopwatch();
            _stopwatchTotal.Start();

            Dictionary<string, string> dict = new Dictionary<string, string>();
            dict.Add("device_id", _deviceId.ToString());
            dict.Add("trt_engine_cache_enable", "1");
            dict.Add("trt_engine_cache_path", Path.Combine(Directory.GetCurrentDirectory(), "trt_engine_cache"));
            dict.Add("trt_engine_cache_prefix", "YoloSharpOnnx");
            dict.Add("trt_auxiliary_streams", "0");
            dict.Add("trt_builder_optimization_level", "3");

            using (YoloSharp yolo = new YoloSharp(new ExecutionProviderTensorRT(modelPath, _deviceId, dict)))
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
            _stopwatchTotal.Stop();

            Console.WriteLine($"time:{_stopwatchTotal.Elapsed}");

        }
    }
}

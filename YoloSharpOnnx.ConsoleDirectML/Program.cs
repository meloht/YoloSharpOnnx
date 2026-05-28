using OpenCvSharp;
using System.Diagnostics;
using System.Runtime.Intrinsics.X86;
using System.Threading.Channels;
using YoloSharpOnnx.DataResult;
using YoloSharpOnnx.Models;
using YoloSharpOnnx.Providers;
using YoloSharpOnnx.TestCommon;
using YoloSharpOnnx.Utils;


namespace YoloSharpOnnx.ConsoleDirectML
{
    internal class Program
    {
        static int _deviceId = 1;
        static string modelPath = @"D:\code\model\best.onnx";
        static string dir = @"D:\code\model\TestImages";
        static void Main(string[] args)
        {
            Console.WriteLine("Hello, World!");

            //TestChannel();

            //TestBatchInfer();
            //TestBatchInferObb();
            //TestBatchInferSeg();
            //TestInferSeg();
            // _ = TestBatchForeachInfer();
            TestInferPerf();
           // _ = TestInferBatchAsync();
            //TestInferCls();
            //TestInfer();
            //_ = Task.Run(async () => await TestInferAsync());

            //TestBufferPool();
            //Task.WaitAll(TestAsyncChannel());
            //Task.WaitAll(TestBatchForeachInfer());
            Console.WriteLine("end!");
            Console.ReadKey();

        }

        private static void TestBufferPool()
        {
            OnnxModel model = new OnnxModel();
            model.InputSizeInBytes = 1280 * 1280 * 3 * sizeof(float);
            model.InputShape = [1, 3, 1280, 1280];

            MatBufferPoolArr bufferPool = new MatBufferPoolArr(10, model);
            ImageBatchData[] arr = new ImageBatchData[20];
            for (int i = 0; i < 20; i++)
            {
                arr[i] = bufferPool.Rent();
            }
            for (int i = 0; i < 20; i++)
            {
                bufferPool.Return(arr[i]);
            }
        }

        private static async Task TestAsyncChannel()
        {

            var model = TestDataUtils.GetModelPath("yolo11n-cls.onnx");
            using YoloSharp yolo = new YoloSharp(new ExecutionProviderDirectML(model, _deviceId));
            string imgPath = TestDataUtils.GetImagePathCls(TestDataUtils.Cls01);

            using IYoloAsync yoloAsync = yolo.CreateAsyncChannel();

            var res = await yoloAsync.RunClassifyAsync(imgPath);
            Console.WriteLine(res.Summary());


            using Mat img = Cv2.ImRead(imgPath);
            var res2 = await yoloAsync.RunClassifyAsync(img);
            string ans2 = res2.Summary();
            Console.WriteLine(ans2);
        }
        private static void TestInferCls()
        {
            string model = @"D:\DemoCode\WinFormsAppYoloCls\WinFormsAppYoloCls\yolo26n-cls.onnx";
            string img = @"D:\code\YoloSharpOnnx\YoloSharpOnnx.TestCommon\TestData\Images\000000000009.jpg";
            using YoloSharp yolo = new YoloSharp(new ExecutionProviderDirectML(model, _deviceId));
            var res = yolo.RunClassifyWithTime(img);
            Console.WriteLine($"{res.ToString()}, {res.SpeedResult.ToString()}");
        }
        private static void TestInfer()
        {
            DirectoryInfo directory = new DirectoryInfo(dir);
            var files = directory.GetFiles();

            System.Diagnostics.Stopwatch _stopwatch = new System.Diagnostics.Stopwatch();
            System.Diagnostics.Stopwatch _stopwatchTotal = new System.Diagnostics.Stopwatch();
            _stopwatchTotal.Start();
            using (YoloSharp yolo = new YoloSharp(new ExecutionProviderDirectML(modelPath, _deviceId)))
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

        private static void TestInferPerf()
        {

            DirectoryInfo directory = new DirectoryInfo(dir);
            var files = directory.GetFiles();
            System.Diagnostics.Stopwatch _stopwatchTotal = new System.Diagnostics.Stopwatch();
            _stopwatchTotal.Start();

            long totalInfer = 0;
            int count = 0;
            Stopwatch stopwatch = new Stopwatch();
            using (YoloSharp yolo = new YoloSharp(new ExecutionProviderDirectML(modelPath, _deviceId)))
            {
                foreach (var item in files)
                {
                    string filePath = item.Extension.ToLower();
                    if (filePath.EndsWith(".jpg") || filePath.EndsWith(".png"))
                    {
                        count++;
                        using Mat img = Cv2.ImRead(item.FullName);
                        var res = yolo.RunDetectWithTime(img);
                        totalInfer += res.SpeedResult.Inference;
                        stopwatch.Restart();
                        yolo.DrawDetections(img, res.Items);
                        stopwatch.Stop();
                        Console.WriteLine($"{item.Name} {res.ToString()}, {res.SpeedResult.ToString()}, draw time: {stopwatch.ElapsedMilliseconds} ms");
                    }
                }
            }

            _stopwatchTotal.Stop();

            float avg = totalInfer / (float)count;
            Console.WriteLine($"total time:{_stopwatchTotal.Elapsed},count:{count} Infer avg time:{avg}ms");

        }

        private static async Task TestInferBatchAsync()
        {
            using var yolo = new YoloSharp(new ExecutionProviderDirectML(modelPath, _deviceId));
            using var yoloAsync = yolo.CreateAsyncChannel();
            var files = Directory.GetFiles(dir);
            count = 1;
            for (int i = 0; i < files.Length; i++)
            {
                using Mat img = Cv2.ImRead(files[i]);
                await yoloAsync.RunDetectAsync(img, Guid.NewGuid(), null, ReceiveProcess);

            }
            await yoloAsync.CompleteAndCloseAsyncChannel();
        }
        static int count = 1;
        private static void ReceiveProcess(DetectAsyncResult e)
        {
            long cost = DateTimeOffset.UtcNow.ToUnixTimeMilliseconds() - e.StartTimestamp;
            string ans = e.Results.Summary();
            Console.WriteLine($"{count++} {ans} time:{cost}ms");

        }

        private static async Task TestInferAsync()
        {

            using var yolo = new YoloSharp(new ExecutionProviderDirectML(modelPath, _deviceId));
            System.Diagnostics.Stopwatch _stopwatchTotal = new System.Diagnostics.Stopwatch();
            _stopwatchTotal.Start();
            var files = Directory.GetFiles(dir);
            yolo.YoloConfiguration.BatchPoolSize = 5;
            using (var yoloAsync = yolo.CreateAsyncChannel())
            {
                for (int i = 0; i < files.Length; i++)
                {

                    var res = await yoloAsync.RunDetectAsync(files[i]);
                    Console.WriteLine($"{i + 1} {YoloUtils.GetDetectResult(res)}");
                }
                await yoloAsync.CompleteAndCloseAsyncChannel();
            }

            _stopwatchTotal.Stop();
            var avg = _stopwatchTotal.ElapsedMilliseconds / files.Length;
            Console.WriteLine($"total time:{_stopwatchTotal.Elapsed}, count:{files.Length} Infer avg time:{avg}ms");

        }
        private static void TestBatchInferObb()
        {
            string modelPath = @"D:\code\model\yolo11n-obb.onnx";
            string dirObb = @"D:\code\model\dota128\images\train";
            DirectoryInfo directory = new DirectoryInfo(dir);
            var files = directory.GetFiles();

            System.Diagnostics.Stopwatch _stopwatch = new System.Diagnostics.Stopwatch();
            int num = files.Length;
            using (YoloSharp yolo = new YoloSharp(new ExecutionProviderDirectML(modelPath, _deviceId)))
            {

                yolo.YoloConfiguration.BatchPoolSize = 80;
        
                _stopwatch.Start();
                var list = yolo.RunBatchObbDetect(dirObb, receiveAction: ReceiveProcess);
                _stopwatch.Stop();

            }


            Console.WriteLine($"detect {num} images, time:{_stopwatch.Elapsed}");
        }

        private static void TestBatchInfer()
        {

            DirectoryInfo directory = new DirectoryInfo(dir);
            var files = directory.GetFiles();

            System.Diagnostics.Stopwatch _stopwatch = new System.Diagnostics.Stopwatch();
            int num = files.Length;
            using (YoloSharp yolo = new YoloSharp(new ExecutionProviderDirectML(modelPath, _deviceId)))
            {

                yolo.YoloConfiguration.BatchPoolSize = 80;
  
                _stopwatch.Start();
                var list = yolo.RunBatchDetect(dir, receiveAction: ReceiveProcess);
                _stopwatch.Stop();


            }


            Console.WriteLine($"detect {num} images, time:{_stopwatch.Elapsed}");
        }

        private static void TestInferSeg()
        {
            System.Diagnostics.Stopwatch _stopwatch = new System.Diagnostics.Stopwatch();
            int num = 0;
            var files = Directory.GetFiles(dir);
            using (YoloSharp yolo = new YoloSharp(new ExecutionProviderDirectML(@"C:\code\model\yolo11n-seg.onnx", _deviceId)))
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

        private static void TestBatchInferSeg()
        {
            System.Diagnostics.Stopwatch _stopwatch = new System.Diagnostics.Stopwatch();
            int num = 0;
            using (YoloSharp yolo = new YoloSharp(new ExecutionProviderDirectML(@"C:\code\model\yolo26n-seg.onnx", _deviceId)))
            {
                yolo.YoloConfiguration.BatchPoolSize = 30;
     
                _stopwatch.Start();
                var list = yolo.RunBatchSegment(@"C:\code\model\coco128-seg\images\train2017", receiveAction: ReceiveProcess);
                num = list.Length;
                _stopwatch.Stop();

            }


            Console.WriteLine($"detect {num} images, time:{_stopwatch.Elapsed}");
        }

        private static async Task TestBatchForeachInfer()
        {
            var files = Directory.GetFiles(dir);
            System.Diagnostics.Stopwatch _stopwatch = new System.Diagnostics.Stopwatch();
            _stopwatch.Start();
            int num = files.Length;
            using (YoloSharp yolo = new YoloSharp(new ExecutionProviderDirectML(modelPath, _deviceId)))
            {
                yolo.YoloConfiguration.BatchPoolSize = 60;

                await foreach (var item in yolo.BatchDetectForeachAsync(files.ToList()))
                {
                    Console.WriteLine($"{item.ImagePath} {YoloUtils.GetDetectResult(item.Results)}");
                }

            }
            _stopwatch.Stop();

            Console.WriteLine($"detect {num} images, time:{_stopwatch.Elapsed}");
        }



        private static void ReceiveProcess(DetectionBatchResult e)
        {


            long cost = DateTimeOffset.UtcNow.ToUnixTimeMilliseconds() - e.StartTimestamp;
            string ans = e.Results.Summary();
            Console.WriteLine($"{e.ImagePath} {ans} time:{cost}ms");

        }
        private static void ReceiveProcess(ObbBatchResult e)
        {

            long cost = DateTimeOffset.UtcNow.ToUnixTimeMilliseconds() - e.StartTimestamp;
            string ans = e.Results.Summary();
            Console.WriteLine($"{e.ImagePath} {ans} time:{cost}ms");

        }
        private static void ReceiveProcess(SegBatchResult e)
        {

            long cost = DateTimeOffset.UtcNow.ToUnixTimeMilliseconds() - e.StartTimestamp;
            string ans = e.Results.Summary();
            Console.WriteLine($"{e.ImagePath} {ans} time:{cost}ms");

        }
        internal class ProcessCallback : IBatchProcessCallback<DetectionBatchResult>
        {

            public void ReceiveProcessResult(DetectionBatchResult e)
            {

                string res = e.Results.Summary();

            }

        }

        public static async Task TestChannel()
        {
            // 1. 创建 有界通道（容量=2）
            Channel<int> channel = Channel.CreateBounded<int>(new BoundedChannelOptions(100)
            {
                // 通道满时的策略：等待（默认，推荐）
                FullMode = BoundedChannelFullMode.Wait
            });

            // 生产者
            var producer = Task.Run(async () =>
            {
                for (int i = 1; i <= 100; i++)
                {
                    await channel.Writer.WriteAsync(i);
                    Console.WriteLine($"生产：{i}");
                    await Task.Delay(10);
                }
                channel.Writer.Complete();
            });

            // 消费者
            var consumer = Task.Run(async () =>
            {
                // 极简读取写法（C# 8+）
                await foreach (var msg in channel.Reader.ReadAllAsync())
                {
                    Console.WriteLine($"消费：{msg}");
                    await Task.Delay(12);
                }
            });

            Task.WaitAll(consumer, producer);


        }
    }
}

namespace YoloSharpOnnx.TestCommon
{
    public class TestDataUtils
    {
        public const string Bus = "bus.jpg";
        public const string Zidane = "zidane.jpg";

        public const string Cls01 = "000000000009.jpg";
        public const string Cls02 = "000000063409.jpg";

        public const string Seg01 = "000000000650.jpg";
        public const string Seg02 = "zidane.jpg";

        public const string Pose01 = "000000000036.jpg";
        public const string Pose02 = "zidane.jpg";

        public const string Obb01 = "P1174__1024__2096___2620.jpg";
        public const string Obb02 = "P1374__682__3490___1396.jpg";


        public static string GetImagePathDetect(string path)
        {
            return Path.Combine(AppContext.BaseDirectory, "TestData", "Images","Detect", path);
        }

        public static string GetImagePathCls(string path)
        {
            return Path.Combine(AppContext.BaseDirectory, "TestData", "Images", "Classify", path);
        }

        public static string GetImagePathSeg(string path)
        {
            return Path.Combine(AppContext.BaseDirectory, "TestData", "Images", "Segment", path);
        }

        public static string GetImagePathPose(string path)
        {
            return Path.Combine(AppContext.BaseDirectory, "TestData", "Images", "Pose", path);
        }

        public static string GetImagePathObb(string path)
        {
            return Path.Combine(AppContext.BaseDirectory, "TestData", "Images", "Obb", path);
        }

        public static string GetModelPath(string path)
        {
            return Path.Combine(AppContext.BaseDirectory, "TestData", "Models", path);
        }
        public static string GetImageDirDetect()
        {
            return Path.Combine(AppContext.BaseDirectory, "TestData", "Images", "Detect");
        }

        public static string GetImageDirCls()
        {
            return Path.Combine(AppContext.BaseDirectory, "TestData", "Images", "Classify");
        }

        public static string GetImageDirSeg()
        {
            return Path.Combine(AppContext.BaseDirectory, "TestData", "Images", "Segment");
        }
        public static string GetImageDirPose()
        {
            return Path.Combine(AppContext.BaseDirectory, "TestData", "Images", "Pose");
        }

        public static string GetImageDirObb()
        {
            return Path.Combine(AppContext.BaseDirectory, "TestData", "Images", "Obb");
        }

        public static Dictionary<string, string> GetYolo11Dict()
        {
            Dictionary<string, string> dict = new Dictionary<string, string>();
            dict.Add(GetImagePathDetect(Bus), Yolo11.Bus);
            dict.Add(GetImagePathDetect(Zidane), Yolo11.Zidane);

            return dict;
        }

        public static Dictionary<string, string> GetYolo11ClsDict()
        {
            Dictionary<string, string> dict = new Dictionary<string, string>();
            dict.Add(GetImagePathCls(Cls01), Yolo11.Cls01);
            dict.Add(GetImagePathCls(Cls02), Yolo11.Cls02);

            return dict;
        }
        public static Dictionary<string, string> GetYolo11PoseDict()
        {
            Dictionary<string, string> dict = new Dictionary<string, string>();
            dict.Add(GetImagePathPose(Pose01), Yolo11.Pose01);
            dict.Add(GetImagePathPose(Pose02), Yolo11.Pose02);

            return dict;
        }

        public static Dictionary<string, string> GetYolo26SegDict()
        {
            Dictionary<string, string> dict = new Dictionary<string, string>();
            dict.Add(GetImagePathSeg(Seg01), Yolo26.Seg01);
            dict.Add(GetImagePathSeg(Seg02), Yolo26.Seg02);

            return dict;
        }
        public static Dictionary<string, string> GetYolo11SegDict()
        {
            Dictionary<string, string> dict = new Dictionary<string, string>();
            dict.Add(GetImagePathSeg(Seg01), Yolo11.Seg01);
            dict.Add(GetImagePathSeg(Seg02), Yolo11.Seg02);

            return dict;
        }

        public static Dictionary<string, string> GetYolo26ObbDict()
        {
            Dictionary<string, string> dict = new Dictionary<string, string>();
            dict.Add(GetImagePathObb(Obb01), Yolo26.Obb01);
            dict.Add(GetImagePathObb(Obb02), Yolo26.Obb02);

            return dict;
        }

        public static List<string> GetImgPaths()
        {
            List<string> list = [GetImagePathDetect(Bus),GetImagePathDetect(Zidane)];
            return list;
        }

        public static List<string> GetImgClsPaths()
        {
            List<string> list = [GetImagePathCls(Cls01), GetImagePathCls(Cls02)];
            return list;
        }

        public static List<string> GetImgSegPaths()
        {
            List<string> list = [GetImagePathSeg(Seg01), GetImagePathSeg(Seg02)];
            return list;
        }

        public static List<string> GetImgPosePaths()
        {
            List<string> list = [GetImagePathPose(Pose01), GetImagePathPose(Pose02)];
            return list;
        }

        public static List<string> GetImgObbPaths()
        {
            List<string> list = [GetImagePathObb(Obb01), GetImagePathObb(Obb02)];
            return list;
        }

    }
}

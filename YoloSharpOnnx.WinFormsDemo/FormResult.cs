using Meloht.WinFormsViewImage;
using OpenCvSharp;
using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Data;
using System.Drawing;
using System.Management;
using System.Text;
using System.Windows.Forms;
using YoloSharpOnnx.DataResult;
using YoloSharpOnnx.Providers;

namespace YoloSharpOnnx.WinFormsDemo
{
    public partial class FormResult : Form
    {
        private string _model;
        private string _image;

        public FormResult(string model, string image)
        {
            InitializeComponent();
            _model = model;
            _image = image;
        }

        private void FormResult_Load(object sender, EventArgs e)
        {
            int deviceId = Utils.GetMainGPU();
            using var yolo = new YoloSharp(new ExecutionProviderDirectML(_model, deviceId));
            string res = string.Empty;
            using Mat image = Cv2.ImRead(_image);
            if (yolo.CurrentModelType == ModelType.ObjectDetection)
            {
                var result = yolo.RunDetectWithTime(_image);
                res = $"{result.ToString()}, {result.SpeedResult.ToString()}";

                yolo.DrawDetections(image, result.Items);
                ShowImageForm(_image, image.ToBytes());
            }
            else if (yolo.CurrentModelType == ModelType.Classification)
            {
                var result = yolo.RunClassifyWithTime(_image);
                res = $"{result.ToString()}, {result.SpeedResult.ToString()}";

                yolo.DrawClassification(image, result.Items);
                ShowImageForm(_image, image.ToBytes());
            }


            this.txtReuslt.Text = res;



        }

        private void ShowImageForm(string path, byte[] image)
        {
            var formView = new FormShowImage(path, image);
            formView.TopLevel = false;

            formView.FormBorderStyle = FormBorderStyle.None;

            // 填充整个 GroupBox
            formView.Dock = DockStyle.Fill;

            // 可选：不显示标题栏图标
            formView.ControlBox = false;

            // 添加到 GroupBox
            groupBoxImage.Controls.Clear();
            groupBoxImage.Controls.Add(formView);

            // 显示
            formView.Show();
        }



    }
}

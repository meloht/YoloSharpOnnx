using Meloht.WinFormsViewImage;
using OpenCvSharp;
using System.Management;
using YoloSharpOnnx.Providers;
using YoloSharpOnnx.TestCommon;
using static System.Net.Mime.MediaTypeNames;

namespace YoloSharpOnnx.WinFormsDemo
{
    public partial class FormTest : Form
    {

        public FormTest()
        {
            InitializeComponent();
            string model = TestDataUtils.GetModelPath("yolo26n-cls.onnx");
            int deviceId = Utils.GetMainGPU();
            using var yolo = new YoloSharp(new ExecutionProviderDirectML(model, deviceId));

            this.textBoxModel.Text = @"D:\code\model\yolo26n-seg.onnx";
            this.textBoxImage.Text = @"D:\code\model\zidane.jpg";
            LoadImageView(this.textBoxImage.Text);
        }

        private void LoadImageView(string fileName)
        {
            var formView = new FormShowImage(fileName);
            formView.TopLevel = false;

            formView.FormBorderStyle = FormBorderStyle.None;

            // 填充整个 GroupBox
            formView.Dock = DockStyle.Fill;

            // 可选：不显示标题栏图标
            formView.ControlBox = false;

            // 添加到 GroupBox
            groupBoxImageView.Controls.Clear();
            groupBoxImageView.Controls.Add(formView);

            // 显示
            formView.Show();

        }

        private void btnSelectModel_Click(object sender, EventArgs e)
        {
            if (openFileDialogModel.ShowDialog() == DialogResult.OK)
            {
                this.textBoxModel.Text = openFileDialogModel.FileName;
            }
        }

        private void btnSelectImage_Click(object sender, EventArgs e)
        {
            try
            {
                if (openFileDialogImage.ShowDialog() == DialogResult.OK)
                {
                    this.textBoxImage.Text = openFileDialogImage.FileName;
                    LoadImageView(this.textBoxImage.Text);
                }
            }
            catch (Exception ex)
            {
                MessageBox.Show(ex.Message, "Error", MessageBoxButtons.OK, MessageBoxIcon.Error);
            }

        }

        private void btnRun_Click(object sender, EventArgs e)
        {
            if (this.textBoxModel.Text.Trim() == string.Empty)
            {
                MessageBox.Show("The model file must be selected", "Warning", MessageBoxButtons.OK, MessageBoxIcon.Warning);
                this.textBoxModel.Focus();
                return;
            }
            if (this.textBoxImage.Text.Trim() == string.Empty)
            {
                MessageBox.Show("The image file must be selected", "Warning", MessageBoxButtons.OK, MessageBoxIcon.Warning);
                this.textBoxImage.Focus();
                return;
            }
            try
            {
                this.btnRunDetect.Enabled = false;
                FormResult form = new FormResult(this.textBoxModel.Text, this.textBoxImage.Text);
                form.Show();

            }
            catch (Exception ex)
            {
                MessageBox.Show(ex.Message, "Error", MessageBoxButtons.OK, MessageBoxIcon.Error);
            }
            finally
            {
                this.btnRunDetect.Enabled = true;
            }


        }







    }
}

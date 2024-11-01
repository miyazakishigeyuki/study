#include <opencv2/opencv.hpp>
#include <iostream>
#include <cmath>

class EquirectangularConverter {
private:
    cv::Mat inputImage;
    int outputWidth, outputHeight;

    // 球面座標への変換
    void cartesianToSpherical(float x, float y, float z, 
                               float& theta, float& phi) {
        theta = std::atan2(y, x);
        phi = std::acos(z);
    }

    // 双一次補間
    cv::Vec3b bilinearInterpolation(const cv::Mat& img, float x, float y) {
        int x0 = static_cast<int>(x);
        int y0 = static_cast<int>(y);
        int x1 = std::min(x0 + 1, img.cols - 1);
        int y1 = std::min(y0 + 1, img.rows - 1);

        float fx = x - x0;
        float fy = y - y0;

        cv::Vec3b p00 = img.at<cv::Vec3b>(y0, x0);
        cv::Vec3b p10 = img.at<cv::Vec3b>(y0, x1);
        cv::Vec3b p01 = img.at<cv::Vec3b>(y1, x0);
        cv::Vec3b p11 = img.at<cv::Vec3b>(y1, x1);

        // 双一次補間の計算
        return cv::Vec3b(
            static_cast<uchar>((1-fx) * (1-fy) * p00[0] + fx * (1-fy) * p10[0] + 
                               (1-fx) * fy * p01[0] + fx * fy * p11[0]),
            static_cast<uchar>((1-fx) * (1-fy) * p00[1] + fx * (1-fy) * p10[1] + 
                               (1-fx) * fy * p01[1] + fx * fy * p11[1]),
            static_cast<uchar>((1-fx) * (1-fy) * p00[2] + fx * (1-fy) * p10[2] + 
                               (1-fx) * fy * p01[2] + fx * fy * p11[2])
        );
    }

public:
    // コンストラクタ
    EquirectangularConverter(const cv::Mat& image, 
                             int outWidth = 800, 
                             int outHeight = 600) 
        : inputImage(image), 
          outputWidth(outWidth), 
          outputHeight(outHeight) {}

    // 透視投影への変換
    cv::Mat convertToPerspective(float fovX = 90.0f, 
                                  float fovY = 90.0f, 
                                  float pitch = 0.0f, 
                                  float yaw = 0.0f) {
        cv::Mat outputImage = cv::Mat::zeros(outputHeight, outputWidth, CV_8UC3);

        float imgWidth = inputImage.cols;
        float imgHeight = inputImage.rows;

        // 角度をラジアンに変換
        float fovXRad = fovX * CV_PI / 180.0f;
        float fovYRad = fovY * CV_PI / 180.0f;
        float pitchRad = pitch * CV_PI / 180.0f;
        float yawRad = yaw * CV_PI / 180.0f;

        #pragma omp parallel for collapse(2)  // 並列処理
        for (int y = 0; y < outputHeight; ++y) {
            for (int x = 0; x < outputWidth; ++x) {
                // 画像中心からの正規化座標
                float nx = (x - outputWidth / 2.0f) / (outputWidth / 2.0f);
                float ny = (y - outputHeight / 2.0f) / (outputHeight / 2.0f);

                // 視野角を考慮した球面座標計算
                float theta = std::atan2(ny, nx) + yawRad;
                float phi = std::sqrt(nx * nx + ny * ny) + pitchRad;

                // 球面座標からデカルト座標への変換
                float z = std::cos(phi);
                float r = std::sin(phi);
                float sx = r * std::cos(theta);
                float sy = r * std::sin(theta);

                // 球面座標から元の画像の座標への逆マッピング
                float lon = std::atan2(sy, sx);
                float lat = std::asin(z);

                float srcX = (lon / CV_PI + 1.0f) / 2.0f * (imgWidth - 1);
                float srcY = (lat / (CV_PI/2.0f) + 1.0f) / 2.0f * (imgHeight - 1);

                // 画像範囲外チェック
                if (srcX >= 0 && srcX < imgWidth && srcY >= 0 && srcY < imgHeight) {
                    outputImage.at<cv::Vec3b>(y, x) = bilinearInterpolation(inputImage, srcX, srcY);
                }
            }
        }

        return outputImage;
    }
};

int main() {
    // 画像の読み込み
    cv::Mat equirectangularImage = cv::imread("equirectangular_image.jpg");

    if (equirectangularImage.empty()) {
        std::cerr << "画像の読み込みに失敗しました！" << std::endl;
        return -1;
    }

    // コンバーターの初期化
    EquirectangularConverter converter(equirectangularImage);

    // 異なる視点での変換
    cv::Mat perspectiveView1 = converter.convertToPerspective(90, 90, 0, 0);     // 正面
    cv::Mat perspectiveView2 = converter.convertToPerspective(90, 90, 30, 45);   // 回転あり
    cv::Mat perspectiveView3 = converter.convertToPerspective(120, 120, 15, 90); // 広角

    // 結果の表示
    cv::imshow("オリジナル (正距円筒投影)", equirectangularImage);
    cv::imshow("透視投影 - 正面", perspectiveView1);
    cv::imshow("透視投影 - 回転", perspectiveView2);
    cv::imshow("透視投影 - 広角", perspectiveView3);

    cv::waitKey(0);
    return 0;
}

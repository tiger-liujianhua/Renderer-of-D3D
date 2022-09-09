#include <vector>
#include <cmath>
#include <iostream>
#include "tgaimage.h"
#include "model.h"
#include "geometry.h"
#include "our_gl.h"

const TGAColor white = TGAColor(255, 255, 255, 255);
const TGAColor red = TGAColor(255, 0, 0, 255);
const TGAColor green = TGAColor(0, 255, 0, 255);

Model* model = NULL;
Vec3f light_dir(0, 1, 1);//��Դλ��
Vec3f eye(1, 0.5, 1.5);
Vec3f center(0, 0, 0);  //����λ��
Vec3f up(0, 1, 0);     //����λ��


const int width = 800;
const int height = 800;
const int depth = 255;

//Vec3f m2v(Matrix m)
//{
//    return Vec3f(m[0][0] / m[3][0], m[1][0] / m[3][0], m[2][0] / m[3][0]);
//}

//Matrix lookat(Vec3f eye, Vec3f center, Vec3f up)
//{
//    //����Z,Ȼ��ͨ��up���X,�����Y
//    Vec3f z = (eye - center).normalize();
//    Vec3f x = (up ^ z).normalize();
//    Vec3f y = (x ^ z).normalize();
//
//    //��ת
//    Matrix rotation = Matrix::identity(4);
//    //ƽ��
//    Matrix translation = Matrix::identity(4);
//
//    //����ֻ����rotation������Ϊ����ά�����Կ�����rotation������ʵ��ƽ�Ʋ���
//
//    for (int i = 0; i < 3; i++)
//    {
//        //ƽ��
//        rotation[i][3] = -eye[i];
//    }
//
//    for (int i = 0; i < 3; i++)
//    {
//        //��ת
//        //��������Ӿ�����ǵ�ǰ������ת����������
//        rotation[0][i] = x[i];
//        rotation[1][i] = up[i];
//        rotation[2][i] = -z[i];
//    }
//    //�����������ת��ƽ�ƣ�Ҳ�����������ƽ�ƺ���ת
//    Matrix res = translation * rotation;
//    return res;
//
//}
//Matrix lookat(Vec3f eye, Vec3f center, Vec3f up) {
//    Vec3f z = (eye - center).normalize();
//    Vec3f x = (up ^ z).normalize();
//    Vec3f y = (z ^ x).normalize();
//    Matrix res = Matrix::identity(4);
//    for (int i = 0; i < 3; i++) {
//        res[0][i] = x[i];
//        res[1][i] = y[i];
//        res[2][i] = z[i];
//        res[i][3] = -center[i];
//    }
//    return res;
//}
//Matrix v2m(Vec3f v)
//{
//    Matrix m(4,1);
//    m[0][0] = v.x;
//    m[1][0] = v.y;
//    m[2][0] = v.z;
//    m[3][0] = 1.f;
//    return m;
//}
//��ͼ����
//Matrix viewport(int x, int y, int w, int h) {
//    Matrix m = Matrix::identity(4);
//    m[0][3] = x + w / 2.f;
//    m[1][3] = y + h / 2.f;
//    m[2][3] = depth / 2.f;
//    m[0][0] = w / 2.f;
//    m[1][1] = h / 2.f;
//    m[2][2] = depth / 2.f;
//    return m;
//}


//Vec3f barycentric(Vec3f A, Vec3f B, Vec3f C, Vec3f P)
//{
//    Vec3f s[2];
//    //����[AB,AC,PA]��x��y����
//    for (int i = 2; i--; ) {
//        s[i][0] = C[i] - A[i];
//        s[i][1] = B[i] - A[i];
//        s[i][2] = A[i] - P[i];
//    }
//    //��[AB,AC,PA]��x��y�������в�ˣ����Եõ���������
//    Vec3f u = s[0]^s[1];
//    if (std::abs(u[2]) > 1e-2) //��1-u-v��u��vȫΪ����0��������ʾ�����������ڲ�
//        return Vec3f(1.f - (u.x + u.y) / u.z, u.y / u.z, u.x / u.z);
//    return Vec3f(-1, 1, 1);
//}

//void line(Vec2i p0,Vec2i p1,TGAImage& image, TGAColor color) {
//    bool steep = false;
//    if (std::abs(p0.x - p1.x) < std::abs(p0.y - p1.y)) {
//        std::swap(p0.x, p0.y);
//        std::swap(p1.x, p1.y);
//        steep = true;
//    }
//    if (p0.x > p1.x) {
//        std::swap(p0.x, p1.x);
//        std::swap(p0.y, p1.y);
//    }
//
//    for (int x = p0.x; x <= p1.x; x++) {
//        float t = (x - p0.x) / (float)(p1.x - p0.x);
//        int y = p0.y * (1. - t) + p1.y * t;
//        if (steep) {
//            image.set(y, x, color);
//        }
//        else {
//            image.set(x, y, color);
//        }
//    }
//}

//void triangle(Vec3f* pts, int* zbuffer, Vec2i *uv, TGAImage& image, float *intensity)
//{
//    //�����Χ��
//    Vec2f bboxmin(std::numeric_limits<float>::max(), std::numeric_limits<float>::max());
//    Vec2f bboxmax(-std::numeric_limits<float>::max(), -std::numeric_limits<float>::max());
//    bboxmin.x = std::min(std::min( pts[0].x, pts[1].x),pts[2].x );
//    bboxmax.x = std::max(std::max( pts[0].x, pts[1].x),pts[2].x );
//    bboxmin.y = std::min(std::min( pts[0].y, pts[1].y),pts[2].y );
//    bboxmax.y = std::max(std::max( pts[0].y, pts[1].y),pts[2].y );
//
//    Vec3i p;
//    for (p.x = bboxmin.x; p.x < bboxmax.x; p.x++)
//    {
//        for (p.y = bboxmin.y; p.y < bboxmax.y; p.y++)
//        {
//            //�ҵ��������겢�ж��Ƿ�����������
//            Vec3f bc_screen = barycentric(pts[0], pts[1], pts[2], p);
//            if (bc_screen.x < 0 || bc_screen.y < 0 || bc_screen.z < 0) continue;
//            p.z = 0;
//            //���������ֵ����UVֵ
//            Vec2i uvp = uv[0] * bc_screen.x + uv[1] * bc_screen.y + uv[2] * bc_screen.z;
//            //ͨ����������������ֵ
//            for (int i = 0; i < 3; i++) p.z += pts[i][2] * bc_screen[i];
//            if (zbuffer[int(p.x + p.y * width)] < p.z)
//            {
//                //�������ֵ
//                zbuffer[int(p.x + p.y * width)] = p.z;
//                TGAColor color = model->diffuse(uvp); //�ҵ���Ӧ����
//                image.set(p.x, p.y, TGAColor(color.r * intensity, color.g * intensity, color.b * intensity, 255));
//            }
//        }
//    }
//
//}
//void triangle(Vec2i* pts, TGAImage& image, TGAColor color)
//{
//    //������Χ�У��������ʼֵ
//    Vec2i bboxmin(image.get_width() - 1, image.get_height() - 1);
//    Vec2i bboxmax(0,0);
//    Vec2i clamp(image.get_width() - 1, image.get_height() - 1);
//    for (int i = 0; i < 3; ++i)
//    {
//        //ȷ����Χ��
//        bboxmin.x = std::max(0, std::min(bboxmin.x, pts[i].x));
//        bboxmin.y = std::max(0, std::min(bboxmin.y, pts[i].y));
//        //��һ��max��minֻ��Ϊ��ȷ����Χ�еĺϷ���
//        bboxmax.x = std::min(clamp.x, std::max(bboxmax.x, pts[i].x));
//        bboxmax.y = std::min(clamp.y, std::max(bboxmax.y, pts[i].y));
//    }
//    Vec2i p;//��ʱ�����Χ�����ÿһ����������
//    for (p.x = bboxmin.x; p.x <= bboxmax.x; p.x++)
//    {
//        for (p.y = bboxmin.y; p.y <= bboxmax.y; p.y++)
//        {
//            //������Χ��
//            //��ȡP����������
//            Vec3f u = barycentric(pts[0], pts[1], pts[2], p);
//            //�ж��Ƿ����������ڣ����ھͲ����������ھͽ�������Ⱦɫ
//            if (u.x < 0 || u.y < 0 || u.z < 0) continue;
//            image.set(p.x, p.y, color);
//        }
//    }
//}

//void triangle(Vec3i t0, Vec3i t1, Vec3i t2, float ity0,float ity1,float ity2, Vec2i uv0,Vec2i uv1,Vec2i uv2, float dis0, float dis1, float dis2, TGAImage& image, int *zbuffer)
//{
//    if (t0.y > t1.y) std::swap(t0, t1);
//    if (t0.y > t2.y) std::swap(t0, t2);
//    if (t1.y > t2.y) std::swap(t1, t2);
//    int total_height = t2.y - t0.y;
//    for (int i = 0; i <= total_height; ++i)
//    {
//        bool second_half = i > t1.y - t0.y || t1.y == t0.y;
//        int segment_height = (second_half ? t2.y - t1.y : t1.y - t0.y) + 1;
//        float alpha = (float)i / total_height;
//        float beta = (float)(second_half?i-(t1.y-t0.y):i) / segment_height;
//        //����A,B���������
//        Vec3i A = t0 + Vec3f(t2 - t0) * alpha;
//        Vec3i B = second_half ? t1 + Vec3f(t2 - t1) * beta : t0 + Vec3f(t1 - t0) * beta;
//        //����A,B����Ĺ���ǿ��
//        float ityA = ity0 + (ity2 - ity0) * alpha;
//        float ityB = second_half ? ity1 + (ity2 - ity1) * beta : ity0 + (ity1 - ity0) * beta;
//        //����UV
//        Vec2i uvA = uv0 + (uv2 - uv0) * alpha;
//        Vec2i uvB = second_half ? uv1 + (uv2 - uv1) * beta : uv0 + (uv1 - uv0) * beta;
//        //�������
//        float disA = dis0 + (dis2 - dis0) * alpha;
//        float disB = second_half ? dis1 + (dis2 - dis1) * beta : dis0 + (dis1 - dis0) * beta;
//        if (A.x > B.x) { std::swap(A, B); std::swap(ityA, ityB); }
//        //x������Ϊѭ������
//        for (int j = A.x; j <= B.x; j++) {
//            float phi = B.x == A.x ? 1. : (float)(j - A.x) / (B.x - A.x);
//            //���㵱ǰ��Ҫ���Ƶ�P�����꣬����ǿ��
//            Vec3i    P = Vec3f(A) + Vec3f(B - A) * phi;
//            float ityP = ityA + (ityB - ityA) * phi;
//            ityP = std::min(1.f, std::abs(ityP) + 0.01f);
//            Vec2i uvP = uvA + (uvB - uvA) * phi;
//            float disP = disA + (disB - disA) * phi;
//            int idx = P.x + P.y * width;
//            //�߽�����
//            if (P.x >= width || P.y >= height || P.x < 0 || P.y < 0) continue;
//            if (zbuffer[idx] < P.z) {
//                zbuffer[idx] = P.z;
//                TGAColor color = model->diffuse(uvP);
//                image.set(P.x, P.y, TGAColor(color.bgra[2], color.bgra[1], color.bgra[0]) * ityP * (20.f / std::pow(disP, 2.f)));
//                //image.set(P.x, P.y, TGAColor(255,255,255)* ityP);
//            }
//        }
//    }
//}

    //for (int y = t0.y;y <= t1.y; y++)
    //{
    //    int segment_height = t1.y - t0.y+1;
    //    float alpha = (float)(y - t0.y) / total_height;
    //    float beta = (float)(y - t0.y) / segment_height;
    //    Vec2i A = t0 + (t2 - t0) * alpha;
    //    Vec2i B = t0 + (t1 - t0) * beta;
    //    if (A.x > B.x)std::swap(A, B);
    //    for (int x = A.x;x <= B.x; x++)
    //    {
    //        image.set(x, y, Color);
    //    }
    //}
    //for (int y = t1.y; y <= t2.y; y++)
    //{
    //    int segment_height = t2.y - t1.y + 1;
    //    float alpha = (float)(y - t0.y) / total_height;
    //    float beta = (float)(y - t1.y) / segment_height;
    //    Vec2i A = t0 + (t2 - t0) * alpha;
    //    Vec2i B = t1 + (t2 - t1) * beta;
    //    if (A.x > B.x)std::swap(A, B);
    //    for (int x = A.x; x <= B.x; x++)
    //    {
    //        image.set(x, y, Color);
    //    }
    //}


//��һ����ֵ�ڵĹ���ǿ�ȸ��滻Ϊһ��
struct ToonShader : public IShader {
    mat<3, 3, float> varying_tri;
    Vec3f          varying_ity;

    virtual ~ToonShader() {}

    virtual Vec4f vertex(int iface, int nthvert) {
        Vec4f gl_Vertex = embed<4>(model->vert(iface, nthvert));
        gl_Vertex = Projection * ModelView * gl_Vertex;
        varying_tri.set_col(nthvert, proj<3>(gl_Vertex / gl_Vertex[3]));

        varying_ity[nthvert] = model->normal(iface, nthvert) * light_dir;

        gl_Vertex = Viewport * gl_Vertex;
        return gl_Vertex;
    }

    virtual bool fragment(Vec3f bar, TGAColor& color) {
        float intensity = varying_ity * bar;
        if (intensity > .85) intensity = 1;
        else if (intensity > .60) intensity = .80;
        else if (intensity > .45) intensity = .60;
        else if (intensity > .30) intensity = .45;
        else if (intensity > .15) intensity = .30;
        color = TGAColor(255, 155, 0) * intensity;
        return false;
    }
};

//�������ɫ��
struct GouraudShader : public IShader {
    //������ɫ���Ὣ����д��varying_intensity
    //ƬԪ��ɫ����varying_intensity�ж�ȡ����
    Vec3f varying_intensity;
    mat<2, 3, float> varying_uv;
    //��������������(����ţ��������)
    virtual Vec4f vertex(int iface, int nthvert) {
        //��������źͶ�����Ŷ�ȡģ�Ͷ�Ӧ���㣬����չΪ4ά 
        Vec4f gl_Vertex = embed<4>(model->vert(iface, nthvert));
        varying_uv.set_col(nthvert, model->uv(iface, nthvert));
        //�任�������굽��Ļ���꣨�ӽǾ���*ͶӰ����*�任����*v��
        //�Ƚ�����ͼ�任��������ռ�ת����������ռ䣬�ٽ���ͶӰ�任������ά�ռ��ɶ�ά�ռ䣬��ͼƬ
        mat<4, 4, float> uniform_M = Projection * ModelView;
        mat<4, 4, float> uniform_MIT = ModelView.invert_transpose();
        gl_Vertex = Viewport * uniform_M * gl_Vertex;
        //�������ǿ�ȣ����㷨����*���շ���
        Vec3f normal = proj<3>(embed<4>(model->normal(iface, nthvert))).normalize();
        varying_intensity[nthvert] = std::max(0.f, model->normal(iface, nthvert) * light_dir); // get diffuse lighting intensity
        return gl_Vertex;
    }
    //���ݴ�����������꣬��ɫ���Լ�varying_intensity�������ǰ���ص���ɫ
    virtual bool fragment(Vec3f bar, TGAColor& color) {
        Vec2f uv = varying_uv * bar;
        TGAColor c = model->diffuse(uv);

        float intensity = varying_intensity * bar;
        //color = TGAColor(255, 255, 255) * intensity;
        color = c * intensity;
        return false; 
    }
};

//����������(����1������2������3���������ǿ��1���������ǿ��2���������ǿ��3��tgaָ�룬zbuffer)
//void triangle(Vec3i t0, Vec3i t1, Vec3i t2, float ity0, float ity1, float ity2, Vec2i uv0, Vec2i uv1, Vec2i uv2, float dis0, float dis1, float dis2, TGAImage& image, int* zbuffer) {
//    //����y�ָ�Ϊ����������
//    if (t0.y == t1.y && t0.y == t2.y) return;
//    if (t0.y > t1.y) { std::swap(t0, t1); std::swap(ity0, ity1); std::swap(uv0, uv1); }
//    if (t0.y > t2.y) { std::swap(t0, t2); std::swap(ity0, ity2); std::swap(uv0, uv2); }
//    if (t1.y > t2.y) { std::swap(t1, t2); std::swap(ity1, ity2); std::swap(uv1, uv2); }
//    int total_height = t2.y - t0.y;
//    for (int i = 0; i < total_height; i++) {
//        bool second_half = i > t1.y - t0.y || t1.y == t0.y;
//        int segment_height = second_half ? t2.y - t1.y : t1.y - t0.y;
//        float alpha = (float)i / total_height;
//        float beta = (float)(i - (second_half ? t1.y - t0.y : 0)) / segment_height;
//        //����A,B���������
//        Vec3i A = t0 + Vec3f(t2 - t0) * alpha;
//        Vec3i B = second_half ? t1 + Vec3f(t2 - t1) * beta : t0 + Vec3f(t1 - t0) * beta;
//        //����A,B����Ĺ���ǿ��
//        float ityA = ity0 + (ity2 - ity0) * alpha;
//        float ityB = second_half ? ity1 + (ity2 - ity1) * beta : ity0 + (ity1 - ity0) * beta;
//        //����UV
//        Vec2i uvA = uv0 + (uv2 - uv0) * alpha;
//        Vec2i uvB = second_half ? uv1 + (uv2 - uv1) * beta : uv0 + (uv1 - uv0) * beta;
//        //�������
//        float disA = dis0 + (dis2 - dis0) * alpha;
//        float disB = second_half ? dis1 + (dis2 - dis1) * beta : dis0 + (dis1 - dis0) * beta;
//        if (A.x > B.x) { std::swap(A, B); std::swap(ityA, ityB); }
//        //x������Ϊѭ������
//        for (int j = A.x; j <= B.x; j++) {
//            float phi = B.x == A.x ? 1. : (float)(j - A.x) / (B.x - A.x);
//            //���㵱ǰ��Ҫ���Ƶ�P�����꣬����ǿ��
//            Vec3i    P = Vec3f(A) + Vec3f(B - A) * phi;
//            float ityP = ityA + (ityB - ityA) * phi;
//            ityP = std::min(1.f, std::abs(ityP) + 0.01f);
//            Vec2i uvP = uvA + (uvB - uvA) * phi;
//            float disP = disA + (disB - disA) * phi;
//            int idx = P.x + P.y * width;
//            //�߽�����
//            if (P.x >= width || P.y >= height || P.x < 0 || P.y < 0) continue;
//            if (zbuffer[idx] < P.z) {
//                zbuffer[idx] = P.z;
//                TGAColor color = model->diffuse(uvP);
//                image.set(P.x, P.y, TGAColor(color.bgra[2], color.bgra[1], color.bgra[0]) * ityP * (20.f / std::pow(disP, 2.f)));
//                //image.set(P.x, P.y, TGAColor(255,255,255)* ityP);
//            }
//        }
//    }
//}

Vec3f world2screen(Vec3f v) {
    return Vec3f(int((v.x + 1.) * width / 2. + .5), int((v.y + 1.) * height / 2. + .5), v.z);
}


int main(int argc, char** argv) {
    //��ȡģ��
    if (2 == argc) {
        model = new Model(argv[1]);
    }
    else {
        model = new Model("obj/african_head.obj");      
    }

    lookat(eye, center, up);                                            //��ͼ�任
    projection(-1.f / (eye - center).norm());                           //ͶӰ�任
    viewport(width / 8, height / 8, width * 3 / 4, height * 3 / 4);     //�ӽǾ���
    light_dir.normalize();                                              //��Դ

    TGAImage image(width, height, TGAImage::RGB);
    TGAImage zbuffer(width, height, TGAImage::GRAYSCALE);

    GouraudShader shader;

        for (int i = 0; i < model->nfaces(); i++) {
            Vec4f screen_coords[3];
            for (int j = 0; j < 3; j++) {
                screen_coords[j] = shader.vertex(i, j);//Ϊ�����ε�ÿ��������ö�����ɫ��
            }
            Vec2i uv[3];
            for (int k = 0; k < 3; k++) {
                uv[k] = model->uv(i, k);
            }
            triangle(screen_coords, shader, image, zbuffer);
        }
        image.flip_vertically();
        image.write_tga_file("output.tga");
   
    delete model;
    return 0;
}

//int main(int argc, char** argv) {
//    //��ȡģ��
//    if (2 == argc) {
//        model = new Model(argv[1]);
//    }
//    else {
//        model = new Model("obj/african_head.obj");
//    }
//    //����zbuffer
//    zbuffer = new int[width * height];
//    for (int i = 0; i < width * height; i++) {
//        //��ʼ��zbuffer
//        zbuffer[i] = std::numeric_limits<int>::min();
//    }
//
//    //����
//    {
//        //��ʼ��͸��ͶӰ����
//        Matrix Projection = Matrix::identity(4);
//        //��ʼ���ӽǾ���
//        Matrix ViewPort = viewport(width / 8, height / 8, width * 3 / 4, height * 3 / 4);
//        //ͶӰ����[3][2]=-1/c��cΪ���z����
//        Projection[3][2] = -1.f / camera.z;
//        //����tga
//        TGAImage image(width, height, TGAImage::RGB);
//        //��ģ����Ϊѭ�����Ʊ���
//        for (int i = 0; i < model->nfaces(); i++) {
//            std::vector<int> face = model->face(i);
//            Vec3f screen_coords[3];
//            Vec3f world_coords[3];
//            float intensity[3];
//            float distance[3];
//            for (int j = 0; j < 3; j++) {
//                Vec3f v = model->vert(face[j]);
//                //�ӽǾ���*ͶӰ����*����
//                Matrix m_v = ViewPort * v2m(v);
//                screen_coords[j] = m2v(ViewPort * Projection *lookat(eye,center,up)* v2m(v));
//                world_coords[j] = v;
//                intensity[j] = model->norm(i, j) * light_dir;
//                Vec3f new_v = m2v(m_v);
//                distance[j] = std::pow((std::pow(new_v.x - eye.x, 2.0f) + std::pow(new_v.y - eye.y, 2.0f) + std::pow(new_v.z - eye.z, 2.0f)), 0.5f);
//            }
//            //���㷨����
//            //Vec3f n = (world_coords[2] - world_coords[0]) ^ (world_coords[1] - world_coords[0]);
//           // n.normalize();
//            //�������
//            //float intensity = n * light_dir;
//            //intensity = std::min(std::abs(intensity), 1.f);
//            Vec2i uv[3];
//            for (int k = 0; k < 3; k++) {
//                uv[k] = model->uv(i, k);
//            }
//                //����������
//                triangle(screen_coords[0], screen_coords[1], screen_coords[2], intensity[0], intensity[1], intensity[2], uv[0], uv[1], uv[2], distance[0], distance[1], distance[2], image, zbuffer);
//            
//        }
//        //tgaĬ��ԭ�������ϣ��ָ�Ϊ����
//        image.flip_vertically();
//        image.write_tga_file("output.tga");
//    }
//    //���zbuffer
//    //{
//    //    TGAImage zbimage(width, height, TGAImage::GRAYSCALE);
//    //    for (int i = 0; i < width; i++) {
//    //        for (int j = 0; j < height; j++) {
//    //            zbimage.set(i, j, TGAColor(zbuffer[i + j * width], 1));
//    //        }
//    //    }
//    //    zbimage.flip_vertically();
//    //    zbimage.write_tga_file("zbuffer.tga");
//    //}
//    delete model;
//    delete[] zbuffer;
//    return 0;
//}


//int main(int argc, char** argv) {
//    if (2 == argc) {
//        model = new Model(argv[1]);
//    }
//    else {
//        model = new Model("obj/african_head.obj");
//    }
//    TGAImage image(width, height, TGAImage::RGB);
//    
//    Vec3f light_dir(0.2, 0.15, -1);
//    //��Ȼ�����������ֵ
//    zbuffer = new int[width * height];
//    for (int i = 0; i < width * height; i++) {
//        //��ʼ��zbuffer
//        zbuffer[i] = std::numeric_limits<int>::min();
//    }
//
//    Matrix Projection = Matrix::identity(4);
//    Matrix ViewPort = viewport(width / 8, height / 8, width * 3 / 4, height * 3 / 4);
//    Projection[3][2] = -1.f / camera.z;
//    for (int i = 0; i < model->nfaces(); i++) {
//        std::vector<int> face = model->face(i);
//        Vec3f pts[3];//��Ļ����
//        Vec3f world_coords[3];
//
//        for (int j = 0; j < 3; ++j)
//        {
//            Vec3f v = model->vert(face[j]);
//            pts[j] = m2v(ViewPort * Projection * v2m(v));
//            world_coords[j] = v;
//        }
//        Vec3f n =(world_coords[2] - world_coords[0])^(world_coords[1] - world_coords[0]);
//        n.normalize();
//        float intensity = n * light_dir;//����ǿ��=������*���շ���   ���������͹��շ����غ�ʱ���������
//        intensity = std::min(std::abs(intensity), 1.f);
//        //ǿ��С��0��˵��ƽ�泯��Ϊ��  ������ü�
//        if (intensity > 0) {
//            Vec2i uv[3];
//            for (int k = 0; k < 3; k++) {
//                uv[k] = model->uv(i, k);//��ȡ���������UVֵ
//            }
//            triangle(pts,zbuffer,uv, image,intensity);
//        }
//        //triangle(pts, zbuffer, image, TGAColor(rand() % 255, rand() % 255, rand() % 255, 255));
//    } 
//  
//    image.flip_vertically(); // i want to have the origin at the left bottom corner of the image
//    image.write_tga_file("output.tga");
//
//    {
//        TGAImage zbimage(width, height, TGAImage::GRAYSCALE);
//        for (int i = 0; i < width; i++) {
//            for (int j = 0; j < height; j++) {
//                zbimage.set(i, j, TGAColor(zbuffer[i + j * width], 1));
//            }
//        }
//        zbimage.flip_vertically();
//        zbimage.write_tga_file("zbuffer.tga");
//    }
//    delete model;
//    return 0;
//}

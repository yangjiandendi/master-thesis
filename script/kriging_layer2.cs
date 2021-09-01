using System.Collections.Generic;
using UnityEngine;
using System.Runtime.CompilerServices;
using NumSharp;
using NumpyDotNet;
using Accord.Math;
using System;
using System.Collections.Generic;
using UnityEngine.UI;

public class kriging_layer2 : MonoBehaviour
{

    public UnityEngine.Vector3 GridSize = new UnityEngine.Vector3(5, 5, 5);
    public float Zoom = 1f;
    public float SurfaceLevel = 0.4f;
    public Material material = null;    
    private GridPoint[,,] p = null;
    private List<UnityEngine.Vector3> vertices = new List<UnityEngine.Vector3>();
    private List<int> triangles = new List<int>();
    private List<Vector2> uv = new List<Vector2>();
    private GridCell cell = new GridCell();
    public static double a_T = 5.0d;
    //private double[][][] intp = new double[5][][];






    private void Start()
    {
        InitGrid();
        BuildMesh();
        //BuildPoints();
    }

    private void Update()
    {
        
        InitGrid();
        BuildMesh();
        var ob2 = GameObject.Find("kriging_model_layer2").GetComponent<Collider>();
        ob2.enabled = false;
        
    }

    //public void AdjustSurfaceLevel(float newSurfaceLevel)
    //{
    //    SurfaceLevel = newSurfaceLevel;
    //}

    private void InitGrid()
    {
        var control_00_pos = GameObject.Find("bezier_surface_point/p00").transform.position;
        var control_01_pos = GameObject.Find("bezier_surface_point/p01").transform.position;
        var control_02_pos = GameObject.Find("bezier_surface_point/p02").transform.position;
        var control_10_pos = GameObject.Find("bezier_surface_point/p10").transform.position;
        var control_11_pos = GameObject.Find("bezier_surface_point/p11").transform.position;
        var control_12_pos = GameObject.Find("bezier_surface_point/p12").transform.position;
        var control_20_pos = GameObject.Find("bezier_surface_point/p20").transform.position;
        var control_21_pos = GameObject.Find("bezier_surface_point/p21").transform.position;
        var control_22_pos = GameObject.Find("bezier_surface_point/p22").transform.position;

        var gradient_pos = GameObject.Find("gradient_point").transform.position;
        var gradient_rot = GameObject.Find("gradient_point").transform.rotation;

        float grad_x = gradient_pos.x;
        float grad_y = gradient_pos.y;
        float grad_z = gradient_pos.z;

        float rot_x = gradient_rot.x;
        float rot_y = gradient_rot.y;
        float rot_z = gradient_rot.z;

        float cp_00_x = control_00_pos.x;
        float cp_00_y = control_00_pos.y;
        float cp_00_z = control_00_pos.z;
        float cp_01_x = control_01_pos.x;
        float cp_01_y = control_01_pos.y;
        float cp_01_z = control_01_pos.z;
        float cp_02_x = control_02_pos.x;
        float cp_02_y = control_02_pos.y;
        float cp_02_z = control_02_pos.z;
        float cp_10_x = control_10_pos.x;
        float cp_10_y = control_10_pos.y;
        float cp_10_z = control_10_pos.z;
        float cp_11_x = control_11_pos.x;
        float cp_11_y = control_11_pos.y;
        float cp_11_z = control_11_pos.z;
        float cp_12_x = control_12_pos.x;
        float cp_12_y = control_12_pos.y;
        float cp_12_z = control_12_pos.z;
        float cp_20_x = control_20_pos.x;
        float cp_20_y = control_20_pos.y;
        float cp_20_z = control_20_pos.z;
        float cp_21_x = control_21_pos.x;
        float cp_21_y = control_21_pos.y;
        float cp_21_z = control_21_pos.z;
        float cp_22_x = control_22_pos.x;
        float cp_22_y = control_22_pos.y;
        float cp_22_z = control_22_pos.z;

        var x1 = cp_00_x;
        var y1 = cp_00_y;
        var z1 = cp_00_z;
        var x2 = cp_10_x;
        var y2 = cp_10_y;
        var z2 = cp_10_z;
        var x3 = cp_20_x;
        var y3 = cp_20_y;
        var z3 = cp_20_z;
        var x4 = cp_01_x;
        var y4 = cp_01_y;
        var z4 = cp_01_z;
        var x5 = cp_11_x;
        var y5 = cp_11_y;
        var z5 = cp_11_z;
        var x6 = cp_21_x;
        var y6 = cp_21_y;
        var z6 = cp_21_z;
        var x7 = cp_02_x;
        var y7 = cp_02_y;
        var z7 = cp_02_z;
        var x8 = cp_12_x;
        var y8 = cp_12_y;
        var z8 = cp_12_z;
        var x9 = cp_22_x;
        var y9 = cp_22_y;
        var z9 = cp_22_z;

        var mp = NumSharp.np.ones(9);

        for (double u = 0; u<=1; u += 0.5)
        {
            for (double v = 0; v<= 1; v += 0.5)
            {
                var x = ((1 - u) * (1 - u)) * ((1 - v) * (1 - v)) * x1 + 2 * u * (1 - u) * ((1 - v) * (1 - v)) * x2 +
                    (u * u) * ((1 - v) * (1 - v)) * x3 + 2 * v * ((1 - u) * (1 - u)) * (1 - v) * x4 +
                     4 * u * v * (1 - u) * (1 - v) * x5 + 2 * (u * u) * v * (1 - v) * x6 +
                     (v * v) * (1 - u) * (1 - u) * x7 + 2 * (v * v) * u * (1 - u) * x8 + (u * u) * (v * v) * x9;

                var y = ((1 - u) * (1 - u)) * ((1 - v) * (1 - v)) * y1 + 2 * u * (1 - u) * ((1 - v) * (1 - v)) * y2 +
                    (u * u) * ((1 - v) * (1 - v)) * y3 + 2 * v * ((1 - u) * (1 - u)) * (1 - v) * y4 +
                     4 * u * v * (1 - u) * (1 - v) * y5 + 2 * (u * u) * v * (1 - v) * y6 +
                     (v * v) * (1 - u) * (1 - u) * y7 + 2 * (v * v) * u * (1 - u) * y8 + (u * u) * (v * v) * y9;

                var z = ((1 - u) * (1 - u)) * ((1 - v) * (1 - v)) * z1 + 2 * u * (1 - u) * ((1 - v) * (1 - v)) * z2 +
                    (u * u) * ((1 - v) * (1 - v)) * z3 + 2 * v * ((1 - u) * (1 - u)) * (1 - v) * z4 +
                     4 * u * v * (1 - u) * (1 - v) * z5 + 2 * (u * u) * v * (1 - v) * z6 +
                     (v * v) * (1 - u) * (1 - u) * z7 + 2 * (v * v) * u * (1 - u) * z8 + (u * u) * (v * v) * z9;

                var p = NumSharp.np.array(new[] { x * x, y * y, z * z, x * y, x * z, y * z, x, y, z });

                mp = NumSharp.np.vstack(mp, p);

            }
        }

        double[][] mp_ = (double[][])mp.ToJaggedArray<double>();
        double[][] mp_T = Accord.Math.Matrix.Transpose<double>(mp_);
        mp_T = Accord.Math.Matrix.RemoveColumn<double>(mp_T, 0);
        mp_ = Accord.Math.Matrix.Transpose<double>(mp_T);
        var pn = NumSharp.np.ones(9, 1);
        double[][] pn_ = (double[][])pn.ToJaggedArray<double>();
        double[][] cof = Accord.Math.Matrix.Solve(mp_, pn_, leastSquares: true);
        

        var bezierxx = cof[0][0];
        var bezieryy = cof[1][0];
        var bezierzz = cof[2][0];
        var bezierxy = cof[3][0];
        var bezierzx = cof[4][0];
        var bezieryz = cof[5][0];
        var bezierx = cof[6][0];
        var beziery = cof[7][0];
        var bezierz = cof[8][0];





        var layer1_point1_pos = GameObject.Find("layer1_point1").transform.position;
        var layer1_point2_pos = GameObject.Find("layer1_point2").transform.position;
        var layer1_point3_pos = GameObject.Find("layer1_point3").transform.position;
        var layer2_point1_pos = GameObject.Find("layer2_point1").transform.position;
        var layer2_point2_pos = GameObject.Find("layer2_point2").transform.position;
        var layer2_point3_pos = GameObject.Find("layer2_point3").transform.position;

        float layer1_x1 = layer1_point1_pos.x;
        float layer1_x2 = layer1_point2_pos.x;
        float layer1_x3 = layer1_point3_pos.x;
        float layer1_y1 = layer1_point1_pos.y;
        float layer1_y2 = layer1_point2_pos.y;
        float layer1_y3 = layer1_point3_pos.y;
        float layer1_z1 = layer1_point1_pos.z;
        float layer1_z2 = layer1_point2_pos.z;
        float layer1_z3 = layer1_point3_pos.z;
        float layer2_x1 = layer2_point1_pos.x;
        float layer2_x2 = layer2_point2_pos.x;
        float layer2_x3 = layer2_point3_pos.x;
        float layer2_y1 = layer2_point1_pos.y;
        float layer2_y2 = layer2_point2_pos.y;
        float layer2_y3 = layer2_point3_pos.y;
        float layer2_z1 = layer2_point1_pos.z;
        float layer2_z2 = layer2_point2_pos.z;
        float layer2_z3 = layer2_point3_pos.z;

        p = new GridPoint[(int)GridSize.x + 1, (int)GridSize.y + 1, (int)GridSize.z + 1];

        var G_1 = NumSharp.np.array(new[,] { { grad_x, grad_y, grad_z } });

        var Rx = NumSharp.np.array(new[,] { {1,0,0},{ 0,Math.Cos(rot_x),-Math.Sin(rot_x) },{ 0, Math.Sin(rot_x) , Math.Cos(rot_x) } }).reshape(3,3);

        var Ry = NumSharp.np.array(new[,] { { Math.Cos(rot_y), 0, Math.Sin(rot_y) },{ 0, 1, 0 },{ -Math.Sin(rot_y),0, Math.Cos(rot_y) } }).reshape(3, 3);

        var Rz = NumSharp.np.array(new[,] { { Math.Cos(rot_z), -Math.Sin(rot_z),0 },{ Math.Sin(rot_z), Math.Cos(rot_z), 0 },{ 0,0,1 } }).reshape(3, 3);

        var G_ini = NumSharp.np.array(new[] { 0,-1,0 }).reshape(3,1);

        var RxRy = NumSharp.np.dot(Rx, Ry);

        var RxRyRz = NumSharp.np.dot(RxRy, Rz);

        var G_1_rot = NumSharp.np.dot(RxRyRz, G_ini);

        double[][] G_1_rot_ = (double[][])G_1_rot.ToJaggedArray<double>();

        var G_1_x = G_1_rot_[0][0];
        var G_1_y = G_1_rot_[1][0];
        var G_1_z = G_1_rot_[2][0];



        //var G_1_x = 0;
        //var G_1_y = 1;
        //var G_1_z = 0;

        var G_1_tiled = NumSharp.np.array(new[] { grad_x, grad_y, grad_z, grad_x, grad_y, grad_z, grad_x, grad_y, grad_z });
        G_1_tiled = G_1_tiled.reshape(3, 3);

        var dist_tiled = Squared_euclidean_distance(G_1_tiled, G_1_tiled);
        var h_u = Cartesian_dist(G_1, G_1);
        var h_v = NumSharp.np.transpose(h_u);

        var a = NumSharp.np.concatenate(new NDArray[] { NumSharp.np.ones(G_1.shape[0], G_1.shape[0]),
                                                   NumSharp.np.zeros(G_1.shape[0], G_1.shape[0]),
                                                   NumSharp.np.zeros(G_1.shape[0], G_1.shape[0]) }, axis: 1);
        var b = NumSharp.np.concatenate(new NDArray[] { NumSharp.np.zeros(G_1.shape[0], G_1.shape[0]),
                                                   NumSharp.np.ones(G_1.shape[0], G_1.shape[0]),
                                                   NumSharp.np.zeros(G_1.shape[0], G_1.shape[0]) }, axis: 1);
        var c = NumSharp.np.concatenate(new NDArray[] { NumSharp.np.zeros(G_1.shape[0], G_1.shape[0]),
                                                   NumSharp.np.zeros(G_1.shape[0], G_1.shape[0]),
                                                   NumSharp.np.ones(G_1.shape[0], G_1.shape[0]) }, axis: 1);

        var perpendicularity_matrix = NumSharp.np.concatenate(new NDArray[] { a, b, c }, axis: 0);

        //var a_T = 5.0;
        var c_o_T = a_T * a_T / 14.0 / 3.0;

        dist_tiled += NumSharp.np.eye(dist_tiled.shape[0]);

        var C_G = Cov_gradients(dist_tiled, a_T, c_o_T, G_1, h_u, h_v);

        var layer1 = NumSharp.np.array(new[,] { { layer1_x1, layer1_y1, layer1_z1 }, { layer1_x2, layer1_y2, layer1_z2 }, { layer1_x3, layer1_y3, layer1_z3 } });
        var layer2 = NumSharp.np.array(new[,] { { layer2_x1, layer2_y1, layer2_z1 }, { layer2_x2, layer2_y2, layer2_z2 }, { layer2_x3, layer2_y3, layer2_z3 } });




        var number_of_layer = 2;
        var number_of_points_per_surface = NumSharp.np.array(new[] { layer1.shape[0], layer2.shape[0] });

        var (ref_layer_points, rest_layer_points) = Set_rest_ref_matrix(number_of_points_per_surface, layer1, layer2);

        var sed_rest_rest = Squared_euclidean_distance(rest_layer_points, rest_layer_points);
        var sed_ref_rest = Squared_euclidean_distance(ref_layer_points, rest_layer_points);
        var sed_rest_ref = Squared_euclidean_distance(rest_layer_points, ref_layer_points);
        var sed_ref_ref = Squared_euclidean_distance(ref_layer_points, ref_layer_points);

        var C_I = Cov_interface(sed_rest_rest, sed_ref_rest, sed_rest_ref, sed_ref_ref);

        var sed_dips_rest = Squared_euclidean_distance(G_1_tiled, rest_layer_points);
        var sed_dips_ref = Squared_euclidean_distance(G_1_tiled, ref_layer_points);

        var hu_rest = Cartesian_dist_no_tile(G_1, rest_layer_points);
        var hu_ref = Cartesian_dist_no_tile(G_1, ref_layer_points);

        var C_GI = Cov_interface_gradients(hu_rest, hu_ref, sed_dips_rest, sed_dips_ref);
        var C_IG = NumSharp.np.transpose(C_GI);

        var K = NumSharp.np.concatenate(new[] { NumSharp.np.concatenate(new[] { C_G, C_GI }, axis: 1), NumSharp.np.concatenate(new[] { C_IG, C_I }, axis: 1) }, axis: 0);

        var xx = NumpyDotNet.np.arange(0, 6, 1.0);
        var yy = NumpyDotNet.np.arange(0, 6, 1.0);
        var zz = NumpyDotNet.np.arange(0, 6, 1.0);
        var XYZ = NumpyDotNet.np.meshgrid(new[] { xx, yy, zz }, copy: true);
        var XX = XYZ[0];
        var YY = XYZ[1];
        var ZZ = XYZ[2];
        var XXX = XX.AsDoubleArray();
        var YYY = YY.AsDoubleArray();
        var ZZZ = ZZ.AsDoubleArray();
        var X = NumSharp.np.array(new[] { XXX });
        var Y = NumSharp.np.array(new[] { YYY });
        var Z = NumSharp.np.array(new[] { ZZZ });
        X = (NumSharp.np.reshape(X, -1)).T;
        Y = (NumSharp.np.reshape(Y, -1)).T;
        Z = (NumSharp.np.reshape(Z, -1)).T;

        var grid = NumSharp.np.stack(new[] { X, Y, Z }, 1);

        var hu_Simpoints = Cartesian_dist_no_tile(G_1, grid);

        var sed_dips_SimPoint = NumSharp.np.sqrt(NumSharp.np.sum(G_1_tiled * G_1_tiled, 1).reshape(G_1_tiled.shape[0], 1) +
                  ((grid * grid)[":,0"] + (grid * grid)[":,1"] + (grid * grid)[":,2"]).reshape(1, grid.shape[0]) -
                  2 * NumSharp.np.matmul(G_1_tiled, NumSharp.np.transpose(grid)));

        var sed_rest_SimPoint = NumSharp.np.sqrt(NumSharp.np.sum(rest_layer_points * rest_layer_points, 1).reshape(rest_layer_points.shape[0], 1) +
                  ((grid * grid)[":,0"] + (grid * grid)[":,1"] + (grid * grid)[":,2"]).reshape(1, grid.shape[0]) -
                  2 * NumSharp.np.matmul(rest_layer_points, NumSharp.np.transpose(grid)));

        var sed_ref_SimPoint = NumSharp.np.sqrt(NumSharp.np.sum(ref_layer_points * ref_layer_points, 1).reshape(ref_layer_points.shape[0], 1) +
                  ((grid * grid)[":,0"] + (grid * grid)[":,1"] + (grid * grid)[":,2"]).reshape(1, grid.shape[0]) -
                  2 * NumSharp.np.matmul(ref_layer_points, NumSharp.np.transpose(grid)));

        //var a1 = NumSharp.np.eye(3);
        //a1[0][0] = bezierxx;
        //a1[1][1] = bezieryy;
        //a1[2][2] = bezierzz;

        //var a2 = NumSharp.np.stack(new[] { ref_layer_points[":,0"] -rest_layer_points[":, 0"],
        //                              ref_layer_points[":, 1"] - rest_layer_points[":, 1"],
        //                              ref_layer_points[":, 2"] - rest_layer_points[":, 2"]});

        //var a3 = NumSharp.np.stack(new[] { NumSharp.np.array(new [] {0 * 2 * bezierxy, 0, 0 }),NumSharp.np.array(new [] {0, 1 * 2 * bezieryz, 0 }),
        //                              NumSharp.np.array(new [] {0, 0, 1 * 2 * bezierzx}),NumSharp.np.array(new [] {0, 1, 0 }) * bezierx,
        //                              NumSharp.np.array(new [] {0, 0, 1 }) * beziery,NumSharp.np.array(new [] {0, 1, 1 }) * bezierz});

        //var a4 = NumSharp.np.stack(new[] { ref_layer_points[":,0"] *ref_layer_points[":, 0"].T - rest_layer_points[":, 0"] * rest_layer_points[":, 0"].T,
        //                              ref_layer_points[":,1"] *ref_layer_points[":, 1"].T - rest_layer_points[":, 1"] * rest_layer_points[":, 1"].T,
        //                              ref_layer_points[":,2"] *ref_layer_points[":, 2"].T - rest_layer_points[":, 2"] * rest_layer_points[":, 2"].T,
        //                              ref_layer_points[":,0"] *ref_layer_points[":, 1"].T - rest_layer_points[":, 0"] * rest_layer_points[":, 1"].T,
        //                              ref_layer_points[":,0"] *ref_layer_points[":, 2"].T - rest_layer_points[":, 0"] * rest_layer_points[":, 2"].T,
        //                              ref_layer_points[":,1"] *ref_layer_points[":, 2"].T - rest_layer_points[":, 1"] * rest_layer_points[":, 2"].T});

        // 0,1,1

        var a1 = 1.0 * bezierx + grad_x * 2 * bezierxx + grad_y * bezierxy + grad_z * bezierzx;

        var a2 = 1.0 * beziery + grad_y * 2 * bezieryy + grad_x * bezierxy + grad_z * bezieryz;

        var a3 = 1.0 * bezierz + grad_z * 2 * bezierzz + grad_x * bezierzx + grad_y * bezieryz;

        var a4 =                      bezierx * (ref_layer_points[":, 0"] - rest_layer_points[":, 0"]) +
                                      beziery * (ref_layer_points[":, 1"] - rest_layer_points[":, 1"]) +
                                      bezierz * (ref_layer_points[":, 2"] - rest_layer_points[":, 2"]) +
                                      bezierxx * (ref_layer_points[":,0"] * ref_layer_points[":, 0"].T - rest_layer_points[":, 0"] * rest_layer_points[":, 0"].T) +
                                      bezieryy * (ref_layer_points[":,1"] * ref_layer_points[":, 1"].T - rest_layer_points[":, 1"] * rest_layer_points[":, 1"].T) +
                                      bezierzz * (ref_layer_points[":,2"] * ref_layer_points[":, 2"].T - rest_layer_points[":, 2"] * rest_layer_points[":, 2"].T) +
                                      bezierxy * (ref_layer_points[":,0"] * ref_layer_points[":, 1"] - rest_layer_points[":, 0"] * rest_layer_points[":, 1"]) +
                                      bezierzx * (ref_layer_points[":,0"] * ref_layer_points[":, 2"] - rest_layer_points[":, 0"] * rest_layer_points[":, 2"]) +
                                      bezieryz * (ref_layer_points[":,1"] * ref_layer_points[":, 2"] - rest_layer_points[":, 1"] * rest_layer_points[":, 2"]);
        var a11 = NumSharp.np.array(new[] { a1, a2, a3 });

        var U = NumSharp.np.concatenate(new[] { a11, a4 }, axis: 0);

        var UU = U.reshape(1,7);
        var U_T = U.reshape(7,1);
        var zero_matrix = NumSharp.np.zeros(1, 1);

        var K_U = NumSharp.np.concatenate(new[] { NumSharp.np.concatenate(new[] { K, U_T }, axis: 1), NumSharp.np.concatenate(new[] { UU, zero_matrix }, axis: 1) }, axis: 0);

        var b_k = NumSharp.np.concatenate(new[] { NumSharp.np.array(new[] { G_1_x, G_1_y, G_1_z }), NumSharp.np.zeros(K_U.shape[0] - G_1.shape[0] * 3) }, axis: 0);
        b_k = b_k.reshape(b_k.shape[0], 1);

        double[][] K_U_ = (double[][])K_U.ToJaggedArray<double>();
        double[][] b_k_ = (double[][])b_k.ToJaggedArray<double>();
        double[][] w_ = Accord.Math.Matrix.Solve(K_U_, b_k_, leastSquares: true);
        var w = NumSharp.np.array(w_);

        var sigma_0_grad = w[":3"] * ((-1.0) * hu_Simpoints * (-c_o_T * (sed_dips_SimPoint < a_T) * ((-14.0 / NumSharp.np.power(a_T, 2)) + 105.0 / 4.0 * sed_dips_SimPoint / NumSharp.np.power(a_T, 3)
                            - 35.0 / 2.0 * NumSharp.np.power(sed_dips_SimPoint, 3) / NumSharp.np.power(a_T, 5) + 21.0 / 4.0 * NumSharp.np.power(sed_dips_SimPoint, 5) / NumSharp.np.power(a_T, 7))));
        
        
        double[][] sigma_0_grad_ = (double[][])sigma_0_grad.ToJaggedArray<double>();
        var sigma_0_grad__ = Accord.Math.Matrix.Sum(sigma_0_grad_, 0);
        sigma_0_grad = NumSharp.np.array(sigma_0_grad__);

        var sigma_0_interf = (-1.0) * w["3:-1"] * (c_o_T * ((1.0 - 7.0 * (sed_rest_SimPoint < a_T) * NumSharp.np.power((sed_rest_SimPoint / a_T), 2) +
                                                                          35.0 / 4.0 * (sed_rest_SimPoint < a_T) * NumSharp.np.power((sed_rest_SimPoint / a_T), 3) -
                                                                          7.0 / 2.0 * (sed_rest_SimPoint < a_T) * NumSharp.np.power((sed_rest_SimPoint / a_T), 5) +
                                                                          3.0 / 4.0 * (sed_rest_SimPoint < a_T) * NumSharp.np.power((sed_rest_SimPoint / a_T), 7)) -
                                                                          (1.0 - 7.0 * (sed_ref_SimPoint < a_T) * NumSharp.np.power((sed_ref_SimPoint / a_T), 2) +
                                                                          35.0 / 4.0 * (sed_ref_SimPoint < a_T) * NumSharp.np.power((sed_ref_SimPoint / a_T), 3) -
                                                                          7.0 / 2.0 * (sed_ref_SimPoint < a_T) * NumSharp.np.power((sed_ref_SimPoint / a_T), 5) +
                                                                          3.0 / 4.0 * (sed_ref_SimPoint < a_T) * NumSharp.np.power((sed_ref_SimPoint / a_T), 7))));
        double[][] sigma_0_interf_ = (double[][])sigma_0_interf.ToJaggedArray<double>();
        var sigma_0_interf__ = Accord.Math.Matrix.Sum(sigma_0_interf_, 0);
        sigma_0_interf = NumSharp.np.array(sigma_0_interf__);


        //var sigma_0_2nd_drift_1 = grid * (w["-9:-6"]).T;
        //double[][] sigma_0_2nd_drift_1_ = (double[][])sigma_0_2nd_drift_1.ToJaggedArray<double>();
        //var sigma_0_2nd_drift_1__ = Accord.Math.Matrix.Sum(sigma_0_2nd_drift_1_, 1);
        //sigma_0_2nd_drift_1 = NumSharp.np.array(sigma_0_2nd_drift_1__);

        var sigma_0_2nd_drift_x = grid[":, 0"] * (w["-1"]).T * bezierx;
        var sigma_0_2nd_drift_y = grid[":, 1"] * (w["-1"]).T * beziery;
        var sigma_0_2nd_drift_z = grid[":, 2"] * (w["-1"]).T * bezierz;
        var sigma_0_2nd_drift_xx = grid[":, 0"] * grid[":, 0"] * (w["-1"]).T * bezierxx;
        var sigma_0_2nd_drift_yy = grid[":, 1"] * grid[":, 1"] * (w["-1"]).T * bezieryy;
        var sigma_0_2nd_drift_zz = grid[":, 2"] * grid[":, 2"] * (w["-1"]).T * bezierzz;
        var sigma_0_2nd_drift_xy = grid[":, 0"] * grid[":, 1"] * (w["-1"]).T * bezierxy;
        var sigma_0_2nd_drift_xz = grid[":, 0"] * grid[":, 2"] * (w["-1"]).T * bezierzx;
        var sigma_0_2nd_drift_yz = grid[":, 1"] * grid[":, 2"] * (w["-1"]).T * bezieryz;

        var sigma_0_2nd_drift = sigma_0_2nd_drift_x+ sigma_0_2nd_drift_y + sigma_0_2nd_drift_z + sigma_0_2nd_drift_xx + sigma_0_2nd_drift_yy +
                                sigma_0_2nd_drift_zz + sigma_0_2nd_drift_xy + sigma_0_2nd_drift_xz + sigma_0_2nd_drift_yz;

        var interpolate_result = sigma_0_grad + sigma_0_interf + sigma_0_2nd_drift;

        var intp = interpolate_result.reshape(6, 6, 6);
        double[][][] intp_ = (double[][][])intp.ToJaggedArray<double>();


        var p1_x = intp_[(int)layer2_y1][(int)layer2_x1 + 1][(int)layer2_z1] * (layer2_x1 - (int)layer2_x1);
        var p1_y = intp_[(int)layer2_y1 + 1][(int)layer2_x1][(int)layer2_z1] * (layer2_y1 - (int)layer2_y1);
        var p1_z = intp_[(int)layer2_y1][(int)layer2_x1][(int)layer2_z1 + 1] * (layer2_z1 - (int)layer2_z1);
        var p1 = (p1_x + p1_y + p1_z) / 3;

        var p2_x = intp_[(int)layer2_y1][(int)layer2_x1][(int)layer2_z1] * (1 - layer2_x1 + (int)layer2_x1);
        var p2_y = intp_[(int)layer2_y1][(int)layer2_x1][(int)layer2_z1] * (1 - layer2_y1 + (int)layer2_y1);
        var p2_z = intp_[(int)layer2_y1][(int)layer2_x1][(int)layer2_z1] * (1 - layer2_z1 + (int)layer2_z1);
        var p2 = (p2_x + p2_y + p2_z) / 3;

        SurfaceLevel = (float)(p1 + p2) + 0.1f ;

        //var p4 = (float)p3;


        //Debug.Log("point1="+ intp_[2][4][1]);
        //Debug.Log("point2=" + intp_[1][1][1]);

        for (int x = 0; x <= GridSize.x; x++)
        {
            for (int y = 0; y <= GridSize.y; y++)
            {
                for (int z = 0; z <= GridSize.z; z++)
                {
                    
                    p[x, y, z] = new GridPoint();
                    p[x, y, z].Position = new UnityEngine.Vector3(x, y, z);
                    p[x, y, z].Value = (float)intp_[y][x][z];
                }
            }
        }

        
    }

    private void BuildMesh()
    {
        GameObject go = this.gameObject;
        MarchingCube.GetMesh(ref go, ref material, true);

        /*  vertex 8 (0-7)
              E4-------------F5         7654-3210
              |               |         HGFE-DCBA
              |               |
        H7-------------G6     |
        |     |         |     |
        |     |         |     |
        |     A0--------|----B1  
        |               |
        |               |
        D3-------------C2               */

        vertices.Clear();
        triangles.Clear();
        uv.Clear();

        for (int z = 0; z < GridSize.z; z++)
        {
            for (int y = 0; y < GridSize.y; y++)
            {
                for (int x = 0; x < GridSize.x; x++)
                {
                    cell.p[0] = p[x, y, z + 1];         //A0
                    cell.p[1] = p[x + 1, y, z + 1];     //B1
                    cell.p[2] = p[x + 1, y, z];         //C2
                    cell.p[3] = p[x, y, z];             //D3
                    cell.p[4] = p[x, y + 1, z + 1];     //E4
                    cell.p[5] = p[x + 1, y + 1, z + 1]; //F5
                    cell.p[6] = p[x + 1, y + 1, z];     //G6
                    cell.p[7] = p[x, y + 1, z];         //H7
                    MarchingCube.IsoFaces(ref cell, SurfaceLevel);
                    BuildMeshCellData(ref cell);
                }
            }
        }

        UnityEngine.Vector3[] av = vertices.ToArray();
        int[] at = triangles.ToArray();
        Vector2[] au = uv.ToArray();
        MarchingCube.SetMesh(ref go, ref av, ref at, ref au);
    }


    private void BuildMeshCellData(ref GridCell cell)
    {
        bool uvAlternate = false;
        for (int i = 0; i < cell.numtriangles; i++)
        {
            vertices.Add(cell.triangle[i].p[0]);
            vertices.Add(cell.triangle[i].p[1]);
            vertices.Add(cell.triangle[i].p[2]);

            //triangles.Add(vertices.Count - 1);  //this order changes side rendered
            //triangles.Add(vertices.Count - 2);
            //triangles.Add(vertices.Count - 3);

            triangles.Add(vertices.Count - 3);  
            triangles.Add(vertices.Count - 2);
            triangles.Add(vertices.Count - 1);

            if (uvAlternate == true)
            {
                uv.Add(UVCoord.A);
                uv.Add(UVCoord.C);
                uv.Add(UVCoord.D);
            }
            else
            {
                uv.Add(UVCoord.A);
                uv.Add(UVCoord.B);
                uv.Add(UVCoord.C);
            }
            uvAlternate = !uvAlternate;
        }
    }
    public static NDArray Squared_euclidean_distance(NDArray x_1, NDArray x_2)
    {
        var sqd = NumSharp.np.sqrt(NumSharp.np.sum(x_1 * x_1, 1).reshape(x_1.shape[0], 1) +
                  NumSharp.np.sum(x_2 * x_2, 1).reshape(1, x_2.shape[0]) -
                  2 * NumSharp.np.matmul(x_1, NumSharp.np.transpose(x_2)));
        return sqd;
    }

    public static NDArray Cartesian_dist(NDArray x_1, NDArray x_2)
    {
        var x = x_1[NumSharp.Slice.All, 0] - x_2[NumSharp.Slice.All, 0].reshape(x_2[NumSharp.Slice.All, 0].shape[0], 1);

        var xx = NumSharp.np.hstack(x, x, x);

        var y = x_1[NumSharp.Slice.All, 1] - x_2[NumSharp.Slice.All, 1].reshape(x_2[NumSharp.Slice.All, 1].shape[0], 1);
        var yy = NumSharp.np.hstack(y, y, y);

        var z = x_1[NumSharp.Slice.All, 2] - x_2[NumSharp.Slice.All, 2].reshape(x_2[NumSharp.Slice.All, 2].shape[0], 1);
        var zz = NumSharp.np.hstack(z, z, z);

        var cd = NumSharp.np.concatenate(new NDArray[] { xx, yy, zz }, axis: 0);

        return cd;
    }


    public static NDArray Cov_gradients(NDArray dist_tiled, double a_T, double c_o_T, NDArray G_1, NDArray h_u, NDArray h_v)
    {
        var condition1 = 0;

        var a = NumSharp.np.concatenate(new NDArray[] { NumSharp.np.ones(G_1.shape[0], G_1.shape[0]),
                                                   NumSharp.np.zeros(G_1.shape[0], G_1.shape[0]),
                                                   NumSharp.np.zeros(G_1.shape[0], G_1.shape[0]) }, axis: 1);
        var b = NumSharp.np.concatenate(new NDArray[] { NumSharp.np.zeros(G_1.shape[0], G_1.shape[0]),
                                                   NumSharp.np.ones(G_1.shape[0], G_1.shape[0]),
                                                   NumSharp.np.zeros(G_1.shape[0], G_1.shape[0]) }, axis: 1);
        var c = NumSharp.np.concatenate(new NDArray[] { NumSharp.np.zeros(G_1.shape[0], G_1.shape[0]),
                                                   NumSharp.np.zeros(G_1.shape[0], G_1.shape[0]),
                                                   NumSharp.np.ones(G_1.shape[0], G_1.shape[0]) }, axis: 1);

        var perpendicularity_matrix = NumSharp.np.concatenate(new NDArray[] { a, b, c }, axis: 0);

        var t1 = NumSharp.np.zeros(3, 3);

        var t2 = -c_o_T * (dist_tiled < a_T)*((-14.0 / NumSharp.np.power(a_T, 2)) +
             105.0 / 4.0 * dist_tiled / NumSharp.np.power(a_T, 3) -
             35.0 / 2.0 * NumSharp.np.power(dist_tiled, 3) / NumSharp.np.power(a_T, 5) +
             21.0 / 4.0 * NumSharp.np.power(dist_tiled, 5) / NumSharp.np.power(a_T, 7)) +
               c_o_T * (dist_tiled < a_T) * 7 * ((9.0 * NumSharp.np.power(dist_tiled, 5) -
                  20.0 * NumSharp.np.power(a_T, 2) * NumSharp.np.power(dist_tiled, 3) +
                  15.0 * NumSharp.np.power(a_T, 4) * dist_tiled -
                  4.0 * NumSharp.np.power(a_T, 5)) / (2.0 * NumSharp.np.power(a_T, 7)));

        var t3 = 
                    c_o_T * (dist_tiled < a_T) * ((-14.0 / NumSharp.np.power(a_T, 2)) + 105.0 / 4.0 * dist_tiled / NumSharp.np.power(a_T, 3) -
                    35.0 / 2.0 * NumSharp.np.power(dist_tiled, 3) / NumSharp.np.power(a_T, 5) +
                    21.0 / 4.0 * NumSharp.np.power(dist_tiled, 5) / NumSharp.np.power(a_T, 7));
        t3 = t3 * perpendicularity_matrix;

        var t4 = 1.0 / 3.0 * NumSharp.np.eye(dist_tiled.shape[0]);

        var C_G = t1 * t2 - t3 + t4;

        return C_G;
    }


    public static (NDArray, NDArray) Set_rest_ref_matrix(NDArray number_of_points_per_surface, NDArray layer1, NDArray layer2)
    {
        var ref_layer_points = NumSharp.np.vstack(NumSharp.np.stack(new[] { layer1[-1], layer1[-1] }, axis: 0), NumSharp.np.stack(new[] { layer2[-1], layer2[-1] }, axis: 0));
        var rest_layer_points = NumSharp.np.concatenate(new[] { layer1["0:-1"], layer2["0:-1"] }, axis: 0);

        return (ref_layer_points, rest_layer_points);
    }

    public static NDArray Cov_interface(NDArray sed_rest_rest, NDArray sed_ref_rest, NDArray sed_rest_ref, NDArray sed_ref_ref)
    {
        //var a_T = 5.0;
        var c_o_T = a_T * a_T / 14.0 / 3.0;

        var C_I = c_o_T * (
            (1.0 - 7.0 * (sed_rest_rest < a_T) * NumSharp.np.power((sed_rest_rest / a_T), 2) +
            35.0 / 4.0 * (sed_rest_rest < a_T) * NumSharp.np.power((sed_rest_rest / a_T), 3) -
            7.0 / 2.0 * (sed_rest_rest < a_T) * NumSharp.np.power((sed_rest_rest / a_T), 5) +
            3.0 / 4.0 * (sed_rest_rest < a_T) * NumSharp.np.power((sed_rest_rest / a_T), 7)) -
            (1.0 - 7.0 * (sed_ref_rest < a_T) * NumSharp.np.power((sed_ref_rest / a_T), 2) +
            35.0 / 4.0 * (sed_ref_rest < a_T) * NumSharp.np.power((sed_ref_rest / a_T), 3) -
            7.0 / 2.0 * (sed_ref_rest < a_T) * NumSharp.np.power((sed_ref_rest / a_T), 5) +
            3.0 / 4.0 * (sed_ref_rest < a_T) * NumSharp.np.power((sed_ref_rest / a_T), 7)) -
            (1.0 - 7.0 * (sed_rest_ref < a_T) * NumSharp.np.power((sed_rest_ref / a_T), 2) +
            35.0 / 4.0 * (sed_rest_ref < a_T) * NumSharp.np.power((sed_rest_ref / a_T), 3) -
            7.0 / 2.0 * (sed_rest_ref < a_T) * NumSharp.np.power((sed_rest_ref / a_T), 5) +
            3.0 / 4.0 * (sed_rest_ref < a_T) * NumSharp.np.power((sed_rest_ref / a_T), 7)) +
            (1.0 - 7.0 * (sed_ref_ref < a_T) * NumSharp.np.power((sed_ref_ref / a_T), 2) +
            35.0 / 4.0 * (sed_ref_ref < a_T) * NumSharp.np.power((sed_ref_ref / a_T), 3) -
            7.0 / 2.0 * (sed_ref_ref < a_T) * NumSharp.np.power((sed_ref_ref / a_T), 5) +
            3.0 / 4.0 * (sed_ref_ref < a_T) * NumSharp.np.power((sed_ref_ref / a_T), 7)));

        return C_I;
    }

    public static NDArray Cartesian_dist_no_tile(NDArray x_1, NDArray x_2)
    {
        var result = NumSharp.np.concatenate(new[] { NumSharp.np.transpose(x_1[":, 0"] - x_2[":, 0"].reshape(x_2.shape[0], 1)),
                                                NumSharp.np.transpose(x_1[":, 1"] - x_2[":, 1"].reshape(x_2.shape[0], 1)),
                                                NumSharp.np.transpose(x_1[":, 2"] - x_2[":, 2"].reshape(x_2.shape[0], 1))  }, axis: 0);

        return result;
    }

    public static NDArray Cov_interface_gradients(NDArray hu_rest, NDArray hu_ref, NDArray sed_dips_rest, NDArray sed_dips_ref)
    {
        //var a_T = 5.0;
        var c_o_T = a_T * a_T / 14.0 / 3.0;

        var C_GI = (hu_rest * (-c_o_T * (sed_dips_rest < a_T) * ((-14.0 / NumSharp.np.power(a_T, 2)) + 105.0 / 4.0 * sed_dips_rest / NumSharp.np.power(a_T, 3) -
                            35.0 / 2.0 * NumSharp.np.power(sed_dips_rest, 3) / NumSharp.np.power(a_T, 5) +
                            21.0 / 4.0 * NumSharp.np.power(sed_dips_rest, 5) / NumSharp.np.power(a_T, 7))) -
                    hu_ref * (-c_o_T * (sed_dips_ref < a_T) * ((-14 / NumSharp.np.power(a_T, 2)) + 105.0 / 4.0 * sed_dips_ref / NumSharp.np.power(a_T, 3) -
                            35.0 / 2.0 * NumSharp.np.power(sed_dips_ref, 3) / NumSharp.np.power(a_T, 5) +
                            21.0 / 4.0 * NumSharp.np.power(sed_dips_ref, 5) / NumSharp.np.power(a_T, 7))));
        return C_GI;

    }

}

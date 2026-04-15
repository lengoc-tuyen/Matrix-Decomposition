from manim import *
import numpy as np

class MasterScene(Scene):
    def construct(self):
        # Cố định random seed 
        np.random.seed(42)

        # Chạy tuần tự các cảnh
        self.scene_01_svd_concept()
        self.scene_02_svd_parallel_transform()
        self.scene_03_svd_decomposition_intro()
        self.scene_04_vt_properties()
        self.scene_05_sigma_properties()
        self.scene_06_u_properties()
        self.scene_07_svd_to_diagonalization()
        self.scene_08_ata_diagonalization()
        self.scene_09_image_matrix_visualizer()
        self.scene_10_svd_compression()

    # ==========================================
    # Cảnh 1: Giới thiệu SVD 
    # ==========================================
    def scene_01_svd_concept(self):
        def create_random_matrix(r, c):
            data = np.random.randint(-9, 10, size=(r, c))
            return Matrix(data)

        # --- 1. Khởi tạo các đối tượng ban đầu ---
        tex_A = MathTex("A")
        tex_eq = MathTex("=")
        label_group = VGroup(tex_A, tex_eq).arrange(RIGHT)
        
        mat_3x3 = create_random_matrix(3, 3)
        mat_2x3 = create_random_matrix(2, 3)
        mat_3x2 = create_random_matrix(3, 2)
        mat_4x4 = create_random_matrix(4, 4)

        eq_group = VGroup(label_group, mat_3x3).arrange(RIGHT).move_to(ORIGIN)

        self.play(Write(label_group), FadeIn(mat_3x3))
        self.wait(1)

        # --- 2. Biến đổi vế phải thành các ma trận kích thước khác nhau ---
        current_mat = mat_3x3
        for next_mat in [mat_2x3, mat_3x2, mat_4x4]:
            temp_label = label_group.copy()
            temp_group = VGroup(temp_label, next_mat).arrange(RIGHT).move_to(ORIGIN)
            
            next_mat.move_to(temp_group[1].get_center())
            
            self.play(
                ReplacementTransform(current_mat, next_mat),
                label_group.animate.move_to(temp_group[0].get_center()),
                run_time=1.2
            )
            current_mat = next_mat
            self.wait(0.6)

        # --- 3. Thu vế phải lại chỉ còn vế trái là A ---
        only_A = MathTex("A").scale(1.5).move_to(ORIGIN)
        
        self.play(
            FadeOut(current_mat), 
            FadeOut(tex_eq),      
            ReplacementTransform(tex_A, only_A), 
            run_time=1.5
        )
        self.wait(1)

        # --- 4. Hiển thị U, Sigma, V^T ở vế phải ---
        formula = MathTex(
            "A", "=", "U", r"\Sigma", "V^T",
            tex_to_color_map={"U": BLUE, r"\Sigma": YELLOW, "V^T": GREEN}
        ).scale(1.5).move_to(ORIGIN)

        self.play(
            ReplacementTransform(only_A, formula[0]),
            Write(formula[1:])
        )
        self.wait(3)
        self.play(*[FadeOut(mob) for mob in self.mobjects])

    # ==========================================
    # Cảnh 2: Mô tả SVD Trên hệ toạ độ
    # ==========================================
    def scene_02_svd_parallel_transform(self):
        # --- ĐỊNH NGHĨA TOÁN HỌC ---
        angle_vt = -30 * DEGREES
        vt = np.array([[np.cos(angle_vt), -np.sin(angle_vt)], [np.sin(angle_vt), np.cos(angle_vt)]])
        
        sigma = np.array([[2, 0], [0, 0.5]])
        
        angle_u = 45 * DEGREES
        u = np.array([[np.cos(angle_u), -np.sin(angle_u)], [np.sin(angle_u), np.cos(angle_u)]])
        
        A = u @ sigma @ vt

        # --- HÀM HỖ TRỢ ---
        def get_plane():
            plane = NumberPlane(
                x_range=[-4, 4, 1], y_range=[-4, 4, 1],
                background_line_style={"stroke_opacity": 0.5}
            )
            circle = Circle(radius=1.0, color=WHITE, stroke_width=3)
            vec_i = Vector([1, 0], color=RED)
            vec_j = Vector([0, 1], color=GREEN)
            return VGroup(plane, circle, vec_i, vec_j).scale(0.6)

        def get_transform_func(matrix, center):
            mat_3x3 = np.eye(3)
            mat_3x3[:2, :2] = matrix 
            def func(p):
                return np.dot(mat_3x3, p - center) + center
            return func

        # --- 1. Thu nhỏ công thức lên trên màn hình ---
        formula = MathTex(
            "A", "=", "U", r"\Sigma", "V^T",
            tex_to_color_map={"U": BLUE, r"\Sigma": YELLOW, "V^T": GREEN}
        ).scale(1.5)
        self.add(formula)
        self.play(formula.animate.scale(0.6).to_edge(UP))
        self.wait(0.5)

        # --- 2. Áp dụng A lên hệ trục tọa độ đầu tiên ---
        plane_left = get_plane().move_to(ORIGIN)
        self.play(Create(plane_left))
        self.wait(1)
        
        text_A = MathTex("A").next_to(plane_left, UP)
        self.play(Write(text_A))
        self.play(
            plane_left.animate.apply_matrix(A), 
            run_time=2
        )
        self.wait(1)

        # --- 3. Làm mờ và di chuyển hệ tọa độ kết quả về bên trái ---
        self.play(
            plane_left.animate.shift(LEFT * 4).set_opacity(0.3),
            text_A.animate.shift(LEFT * 4).set_opacity(0.3),
            run_time=1.5
        )

        # --- 4. Thêm quy trình SVD ở giữa màn hình ---
        step1 = MathTex(r"\text{Rotation } (V^T)", color=GREEN).scale(0.7)
        arrow1 = MathTex(r"\downarrow")
        step2 = MathTex(r"\text{Scaling } (\Sigma)", color=YELLOW).scale(0.7)
        arrow2 = MathTex(r"\downarrow")
        step3 = MathTex(r"\text{Rotation } (U)", color=BLUE).scale(0.7)

        pipeline = VGroup(step1, arrow1, step2, arrow2, step3).arrange(DOWN, buff=0.3)
        self.play(Write(pipeline))
        self.wait(1)

        # --- 5. Vẽ trục tọa độ mới ở bên phải ---
        plane_right = get_plane().shift(RIGHT * 4)
        self.play(Create(plane_right))
        self.wait(1)

        # --- 6. Lần lượt thực hiện 3 bước trên trục bên phải ---
        center_right = plane_right.get_center()

        self.play(Indicate(step1, scale_factor=1.2))
        self.play(ApplyPointwiseFunction(
            get_transform_func(vt, center_right), plane_right
        ), run_time=2)
        self.wait(2) 

        self.play(Indicate(step2, scale_factor=1.2))
        self.play(ApplyPointwiseFunction(
            get_transform_func(sigma, center_right), plane_right
        ), run_time=2)
        self.wait(2) 

        self.play(Indicate(step3, scale_factor=1.2))
        self.play(ApplyPointwiseFunction(
            get_transform_func(u, center_right), plane_right
        ), run_time=2)
        self.wait(2) 

        # --- 7. Gộp 2 hệ tọa độ lại làm 1 ở giữa màn hình ---
        self.play(FadeOut(pipeline), FadeOut(text_A))
        self.play(
            plane_left.animate.move_to(ORIGIN).set_opacity(1),
            plane_right.animate.move_to(ORIGIN),
            run_time=2
        )
        self.wait(2)
        
        self.play(Indicate(plane_right, color=WHITE, scale_factor=1.05))
        self.wait(2)
        self.play(*[FadeOut(mob) for mob in self.mobjects])

    # ==========================================
    # Cảnh 3: Mô tả một trường hợp thử nghiệm
    # ==========================================
    def scene_03_svd_decomposition_intro(self):
        # --- 1. Khởi tạo Ma trận A ---
        tex_A = MathTex("A")
        tex_eq = MathTex("=")
        mat_A = Matrix([[12, 11], [4, 12]], element_to_mobject_config={"color": WHITE})
        group_A = VGroup(tex_A, tex_eq, mat_A).arrange(RIGHT, buff=0.3).move_to(ORIGIN)
        
        self.play(FadeIn(group_A, shift=UP))
        self.wait(1.5)

        # --- 2. Khởi tạo các ma trận thành phần ---
        u_coeff = MathTex(r"\frac{1}{5}", color=BLUE)
        u_mat = Matrix([["4", "-3"], ["3", "4"]]).scale(0.9)
        u_mat.get_brackets().set_color(BLUE)
        group_u = VGroup(u_coeff, u_mat).arrange(RIGHT, buff=0.1)

        sig_mat = Matrix([["20", "0"], ["0", "5"]]).scale(0.9)
        sig_mat.get_brackets().set_color(YELLOW)

        vt_coeff = MathTex(r"\frac{1}{5}", color=GREEN)
        vt_mat = Matrix([["3", "4"], ["-4", "3"]]).scale(0.9)
        vt_mat.get_brackets().set_color(GREEN)
        group_vt = VGroup(vt_coeff, vt_mat).arrange(RIGHT, buff=0.1)

        group_RHS = VGroup(group_u, sig_mat, group_vt).arrange(RIGHT, buff=0.2)
        full_equation = VGroup(tex_A, tex_eq, group_RHS).arrange(RIGHT, buff=0.3).move_to(ORIGIN)

        # --- 3. Hoạt ảnh tách ma trận ---
        copy_A_for_U = mat_A.copy()
        copy_A_for_Sig = mat_A.copy()
        copy_A_for_VT = mat_A.copy()

        self.play(
            tex_A.animate.move_to(full_equation[0].get_center()),
            tex_eq.animate.move_to(full_equation[1].get_center()),
            FadeOut(mat_A), 
            Transform(copy_A_for_U, group_u),
            Transform(copy_A_for_Sig, sig_mat),
            Transform(copy_A_for_VT, group_vt),
            run_time=2
        )
        self.wait(0.5)

        # --- 4. Hiển thị ký hiệu ở dưới ---
        label_U = MathTex("U", color=BLUE).next_to(copy_A_for_U, DOWN, buff=0.5)
        label_Sig = MathTex(r"\Sigma", color=YELLOW).next_to(copy_A_for_Sig, DOWN, buff=0.5)
        label_VT = MathTex("V^T", color=GREEN).next_to(copy_A_for_VT, DOWN, buff=0.5)

        self.play(
            Write(label_U),
            Write(label_Sig),
            Write(label_VT)
        )
        self.wait(3)
        self.play(*[FadeOut(mob) for mob in self.mobjects])

    # ==========================================
    # Cảnh 4: Tính chất của Ma trận V^T
    # ==========================================
    def scene_04_vt_properties(self):
        # --- 1. Chuyển tiếp công thức SVD ---
        formula = MathTex(
            "A", "=", "U", r"\Sigma", "V^T",
            tex_to_color_map={"U": BLUE, r"\Sigma": YELLOW, "V^T": GREEN}
        ).scale(1.5)
        
        self.play(Write(formula))
        self.wait(1)

        self.play(formula.animate.scale(0.6).to_edge(UP))
        self.wait(0.5)

        # --- 2. Hiển thị Ma trận V^T ---
        label_vt = MathTex("V^T_{n \\times n} =", color=GREEN)
        vt_coeff = MathTex(r"\frac{1}{5}", color=GREEN)
        
        vt_mat = Matrix([["3", "4"], ["-4", "3"]], element_to_mobject_config={"color": WHITE})
        vt_mat.get_brackets().set_color(GREEN)
        
        group_vt = VGroup(label_vt, vt_coeff, vt_mat).arrange(RIGHT, buff=0.2)
        group_vt.move_to(UP * 1.5) 

        self.play(FadeIn(group_vt, shift=DOWN))
        self.wait(1)

        # --- 3. Kéo các vector hàng ra ---
        row1_rect = SurroundingRectangle(vt_mat.get_rows()[0], color=RED, buff=0.1)
        v1_label = MathTex("v_1^T", color=RED).next_to(row1_rect, RIGHT)
        
        self.play(Create(row1_rect), Write(v1_label))
        self.wait(0.5)

        row2_rect = SurroundingRectangle(vt_mat.get_rows()[1], color=BLUE, buff=0.1)
        v2_label = MathTex("v_2^T", color=BLUE).next_to(row2_rect, RIGHT)

        self.play(Create(row2_rect), Write(v2_label))
        self.wait(1)

        # --- 4. Chứng minh bằng ký hiệu toán học ---
        calc_norm = MathTex(
            "||", "v_1", "||", "=", 
            r"\sqrt{\left(\frac{3}{5}\right)^2 + \left(\frac{4}{5}\right)^2}", "=", 
            r"\sqrt{\frac{25}{25}}", "=", "1"
        )
        calc_norm[1].set_color(RED) 

        calc_dot = MathTex(
            "v_1", r"\cdot", "v_2", "=", 
            r"\left(\frac{3}{5}\right)\left(\frac{-4}{5}\right) + \left(\frac{4}{5}\right)\left(\frac{3}{5}\right)", "=", 
            r"-\frac{12}{25} + \frac{12}{25}", "=", "0"
        )
        calc_dot[0].set_color(RED)  
        calc_dot[2].set_color(BLUE) 

        calculations = VGroup(calc_norm, calc_dot).arrange(DOWN, buff=1).shift(DOWN * 1.2)

        self.play(Write(calc_norm[0:4])) 
        self.wait(0.5)
        self.play(Write(calc_norm[4]))   
        self.wait(0.5)
        self.play(Write(calc_norm[5:7])) 
        self.wait(0.5)
        self.play(Write(calc_norm[7:]))  
        
        box_1 = SurroundingRectangle(calc_norm[-1], color=YELLOW, buff=0.1)
        self.play(Create(box_1))
        self.wait(1.5)

        self.play(Write(calc_dot[0:4]))  
        self.wait(0.5)
        self.play(Write(calc_dot[4]))    
        self.wait(0.5)
        self.play(Write(calc_dot[5:7]))  
        self.wait(0.5)
        self.play(Write(calc_dot[7:]))   
        
        box_0 = SurroundingRectangle(calc_dot[-1], color=YELLOW, buff=0.1)
        self.play(Create(box_0))
        self.wait(2)

        self.play(*[FadeOut(mob) for mob in self.mobjects])
        self.wait(1)

    # ==========================================
    # Cảnh 5: Tính chất ma trận Sigma
    # ==========================================
    def scene_05_sigma_properties(self):
        # --- 1. Chuyển tiếp công thức SVD ---
        formula = MathTex(
            "A", "=", "U", r"\Sigma", "V^T",
            tex_to_color_map={"U": BLUE, r"\Sigma": YELLOW, "V^T": GREEN}
        ).scale(1.5)
        
        self.play(Write(formula))
        self.wait(1)

        self.play(formula.animate.scale(0.6).to_edge(UP))
        self.wait(0.5)

        # --- 2. Hiển thị Ma trận Sigma ---
        label_sig = MathTex(r"\Sigma_{m \times n} =", color=YELLOW)
        
        sig_mat = Matrix(
            [["20", "0"], ["0", "5"]],
            element_to_mobject_config={"color": WHITE}
        )
        sig_mat.get_brackets().set_color(YELLOW)
        
        group_sig = VGroup(label_sig, sig_mat).arrange(RIGHT, buff=0.3).scale(1.2)
        group_sig.move_to(UP * 1.2) 
        
        self.play(FadeIn(group_sig, shift=DOWN))
        self.wait(1)

        # --- 3. Minh họa đường chéo (Diagonal) ---
        start_pos = sig_mat.get_entries()[0].get_center()
        end_pos = sig_mat.get_entries()[3].get_center()
        
        diag_line = Line(
            start_pos + UP*0.4 + LEFT*0.4, 
            end_pos + DOWN*0.4 + RIGHT*0.4, 
            color=YELLOW, 
            stroke_width=3
        )
        
        self.play(Create(diag_line))
        self.wait(1) 
        self.play(FadeOut(diag_line)) 

        # --- 4. Làm sáng các giá trị 0 ---
        zero_1 = sig_mat.get_entries()[1]
        zero_2 = sig_mat.get_entries()[2]
        
        self.play(
            zero_1.animate.set_color(GRAY_B).scale(1.2),
            zero_2.animate.set_color(GRAY_B).scale(1.2),
        )
        self.play(Indicate(zero_1, color=WHITE, scale_factor=1.5), 
                  Indicate(zero_2, color=WHITE, scale_factor=1.5))
        self.wait(1)

        # --- 5. Ký hiệu giá trị đặc dị ---
        sigma_1_label = MathTex(r"\sigma_1", color=YELLOW).next_to(sig_mat.get_entries()[0], UP, buff=0.4)
        sigma_2_label = MathTex(r"\sigma_2", color=YELLOW).next_to(sig_mat.get_entries()[3], DOWN, buff=0.4)
        
        condition = MathTex(
            r"\sigma_1 \ge \sigma_2 \ge \dots \ge 0",
            color=YELLOW
        ).scale(1.2).shift(DOWN * 1.5)

        self.play(Write(sigma_1_label), Write(sigma_2_label))
        self.wait(0.5)
        self.play(Write(condition))
        
        box_positive = SurroundingRectangle(condition, color=YELLOW, buff=0.2)
        self.play(Create(box_positive))
        self.wait(2)

        self.play(*[FadeOut(mob) for mob in self.mobjects])
        self.wait(1)

    # ==========================================
    # Cảnh 6: Tính chất ma trận U
    # ==========================================
    def scene_06_u_properties(self):
        # --- 1. Chuyển tiếp công thức SVD ---
        formula = MathTex(
            "A", "=", "U", r"\Sigma", "V^T",
            tex_to_color_map={"U": BLUE, r"\Sigma": YELLOW, "V^T": GREEN}
        ).scale(1.5)
        
        self.play(Write(formula))
        self.wait(1)

        self.play(formula.animate.scale(0.6).to_edge(UP))
        self.wait(0.5)

        # --- 2. Hiển thị Ma trận U ---
        label_u = MathTex("U_{m \\times m} =", color=BLUE)
        u_coeff = MathTex(r"\frac{1}{5}", color=BLUE)
        
        u_mat = Matrix([["4", "-3"], ["3", "4"]], element_to_mobject_config={"color": WHITE})
        u_mat.get_brackets().set_color(BLUE)
        
        group_u = VGroup(label_u, u_coeff, u_mat).arrange(RIGHT, buff=0.2)
        group_u.move_to(UP * 1.5)

        self.play(FadeIn(group_u, shift=DOWN))
        self.wait(1)

        # --- 3. Kéo các vector cột ---
        col1_rect = SurroundingRectangle(u_mat.get_columns()[0], color=TEAL, buff=0.1)
        u1_label = MathTex("u_1", color=TEAL).next_to(col1_rect, DOWN, buff=0.3)
        
        self.play(Create(col1_rect), Write(u1_label))
        self.wait(0.5)

        col2_rect = SurroundingRectangle(u_mat.get_columns()[1], color=PURPLE_B, buff=0.1)
        u2_label = MathTex("u_2", color=PURPLE_B).next_to(col2_rect, DOWN, buff=0.3)

        self.play(Create(col2_rect), Write(u2_label))
        self.wait(1)

        # --- 4. Chứng minh bằng toán học ---
        calc_norm = MathTex(
            "||", "u_1", "||", "=", 
            r"\sqrt{\left(\frac{4}{5}\right)^2 + \left(\frac{3}{5}\right)^2}", "=", 
            r"\sqrt{\frac{25}{25}}", "=", "1"
        )
        calc_norm[1].set_color(TEAL) 

        calc_dot = MathTex(
            "u_1", r"\cdot", "u_2", "=", 
            r"\left(\frac{4}{5}\right)\left(\frac{-3}{5}\right) + \left(\frac{3}{5}\right)\left(\frac{4}{5}\right)", "=", 
            r"-\frac{12}{25} + \frac{12}{25}", "=", "0"
        )
        calc_dot[0].set_color(TEAL)    
        calc_dot[2].set_color(PURPLE_B) 

        calculations = VGroup(calc_norm, calc_dot).arrange(DOWN, buff=0.9).shift(DOWN * 1.7)

        self.play(Write(calc_norm[0:4])) 
        self.wait(0.5)
        self.play(Write(calc_norm[4]))   
        self.wait(0.5)
        self.play(Write(calc_norm[5:7])) 
        self.wait(0.5)
        self.play(Write(calc_norm[7:]))  
        
        box_1 = SurroundingRectangle(calc_norm[-1], color=YELLOW, buff=0.1)
        self.play(Create(box_1))
        self.wait(1.5)

        self.play(Write(calc_dot[0:4]))  
        self.wait(0.5)
        self.play(Write(calc_dot[4]))    
        self.wait(0.5)
        self.play(Write(calc_dot[5:7]))  
        self.wait(0.5)
        self.play(Write(calc_dot[7:]))   
        
        box_0 = SurroundingRectangle(calc_dot[-1], color=YELLOW, buff=0.1)
        self.play(Create(box_0))
        self.wait(2)

        self.play(*[FadeOut(mob) for mob in self.mobjects])
        self.wait(1)

    # ==========================================
    # Cảnh 7: Công thức chéo hoá của SVD
    # ==========================================
    def scene_07_svd_to_diagonalization(self):
        scale_factor = 1.3 

        # --- 1. Xuất hiện ma trận SVD ---
        eq_A = MathTex("A", "=", "U", r"\Sigma", "V^T").scale(scale_factor)
        eq_A[2].set_color(BLUE)
        eq_A[3].set_color(YELLOW)
        eq_A[4].set_color(GREEN)
        
        self.play(Write(eq_A))
        self.wait(1)
        self.play(eq_A.animate.shift(UP * 2.5))
        
        # --- 2. Biến đổi từ A thành A^T ---
        eq_AT = MathTex("A^T", "=", "V", r"\Sigma", "U^T").scale(scale_factor)
        eq_AT[2].set_color(GREEN)
        eq_AT[3].set_color(YELLOW)
        eq_AT[4].set_color(BLUE)
        eq_AT.next_to(eq_A, DOWN, buff=0.8)
        
        part_A = eq_A[0].copy()
        part_eq = eq_A[1].copy()
        part_U = eq_A[2].copy()
        part_Sig = eq_A[3].copy()
        part_VT = eq_A[4].copy()

        self.play(
            Transform(part_A, eq_AT[0]),
            Transform(part_eq, eq_AT[1]),
            Transform(part_VT, eq_AT[2], path_arc=-0.8), 
            Transform(part_Sig, eq_AT[3]),
            Transform(part_U, eq_AT[4], path_arc=-0.8), 
            run_time=2
        )
        self.wait(1)
        
        self.add(eq_AT)
        self.remove(part_A, part_eq, part_U, part_Sig, part_VT)

        # --- 3. Thực hiện nhân A^T với A ---
        exp1 = MathTex("A^T A", "=", "V", r"\Sigma", "U^T", "U", r"\Sigma", "V^T").scale(scale_factor)
        exp1[2].set_color(GREEN)
        exp1[3].set_color(YELLOW)
        exp1[4].set_color(BLUE)   
        exp1[5].set_color(BLUE)   
        exp1[6].set_color(YELLOW)
        exp1[7].set_color(GREEN)
        exp1.next_to(eq_AT, DOWN, buff=0.8)
        
        self.play(Write(exp1))
        self.wait(1)

        brace = Brace(VGroup(exp1[4], exp1[5]), DOWN, color=WHITE)
        label_I_brace = MathTex("I").scale(scale_factor).next_to(brace, DOWN)
        
        self.play(Create(brace), Write(label_I_brace))
        self.wait(1)

        exp2 = MathTex("A^T A", "=", "V", r"\Sigma", "I", r"\Sigma", "V^T").scale(scale_factor)
        exp2[2].set_color(GREEN)
        exp2[3].set_color(YELLOW)
        exp2[4].set_color(WHITE) 
        exp2[5].set_color(YELLOW)
        exp2[6].set_color(GREEN)
        exp2.move_to(exp1.get_center())

        self.play(
            ReplacementTransform(exp1[0:4], exp2[0:4]),
            ReplacementTransform(VGroup(exp1[4:6], brace, label_I_brace), exp2[4]), 
            ReplacementTransform(exp1[6:], exp2[5:]),
        )
        self.wait(1)

        exp3 = MathTex("A^T A", "=", "V", r"\Sigma", r"\Sigma", "V^T").scale(scale_factor)
        exp3[2].set_color(GREEN)
        exp3[3].set_color(YELLOW)
        exp3[4].set_color(YELLOW)
        exp3[5].set_color(GREEN)
        exp3.move_to(exp2.get_center())

        self.play(
            ReplacementTransform(exp2[0:4], exp3[0:4]),
            FadeOut(exp2[4], scale=0.5), 
            ReplacementTransform(exp2[5:], exp3[4:])
        )
        self.wait(1)

        eq_ATA = MathTex("A^T A", "=", "V", r"\Sigma^2", "V^T").scale(scale_factor)
        eq_ATA[2].set_color(GREEN)
        eq_ATA[3].set_color(YELLOW)
        eq_ATA[4].set_color(GREEN)
        eq_ATA.move_to(exp3.get_center())

        self.play(
            ReplacementTransform(exp3[0:3], eq_ATA[0:3]),
            ReplacementTransform(exp3[3:5], eq_ATA[3]), 
            ReplacementTransform(exp3[5], eq_ATA[4])
        )
        
        self.wait(2) 
        
        # --- 4. Suy luận V^T = V^-1 ---
        note_V = MathTex("V^T = V^{-1}", color=GREEN).scale(scale_factor)
        note_V.next_to(eq_ATA, RIGHT, buff=1.0)
        
        self.play(Write(note_V))
        self.wait(1)
        
        # --- 5. Thế vào công thức A^T A ---
        eq_ATA_final = MathTex("A^T A", "=", "V", r"\Sigma^2", "V^{-1}").scale(scale_factor)
        eq_ATA_final[2].set_color(GREEN)
        eq_ATA_final[3].set_color(YELLOW)
        eq_ATA_final[4].set_color(GREEN)
        eq_ATA_final.move_to(eq_ATA.get_center())
        
        self.play(
            ReplacementTransform(eq_ATA, eq_ATA_final),
            Indicate(note_V, scale_factor=1.2)
        )
        self.wait(1)
        
        self.play(
            FadeOut(eq_A), FadeOut(eq_AT), FadeOut(note_V),
            eq_ATA_final.animate.move_to(UP * 1.5)
        )
        self.wait(0.5)

        # --- 6. Xuất hiện công thức chéo hóa ---
        eq_Diag = MathTex("A", "=", "P", "D", "P^{-1}").scale(scale_factor)
        eq_Diag.next_to(eq_ATA_final, DOWN, buff=1.5)
        
        self.play(Write(eq_Diag))
        self.wait(1)
        
        # --- 7. Hiệu ứng nhảy lên ---
        def jump_pair(obj1, obj2, highlight_color):
            self.play(
                obj1.animate.shift(UP * 0.3), obj2.animate.shift(UP * 0.3), run_time=0.2
            )
            self.play(
                obj1.animate.shift(DOWN * 0.3), obj2.animate.shift(DOWN * 0.3), run_time=0.2
            )
            self.play(
                Indicate(obj1, color=highlight_color, scale_factor=1.2), 
                Indicate(obj2, color=highlight_color, scale_factor=1.2)
            )

        jump_pair(eq_Diag[2], eq_ATA_final[2], GREEN)   
        self.wait(0.5)
        
        jump_pair(eq_Diag[3], eq_ATA_final[3], YELLOW)  
        self.wait(0.5)
        
        jump_pair(eq_Diag[4], eq_ATA_final[4], GREEN)   
        self.wait(2)

        self.play(*[FadeOut(mob) for mob in self.mobjects])
        self.wait(1)

    # ==========================================
    # Cảnh 8: Thực hiện chéo hoá ma trận ví dụ
    # ==========================================
    def scene_08_ata_diagonalization(self):
        def get_matrix(data, color=WHITE):
            mat = Matrix(data, element_to_mobject_config={"color": WHITE})
            mat.get_brackets().set_color(color)
            return mat

        # --- 1. Tính A^T A với số liệu mẫu ---
        eq_1 = MathTex("A^T A", "=")
        mat_AT = get_matrix([["12", "4"], ["11", "12"]])
        mat_A = get_matrix([["12", "11"], ["4", "12"]])
        
        group_calc = VGroup(eq_1, mat_AT, mat_A).arrange(RIGHT).shift(UP * 1.5)

        self.play(Write(group_calc))
        self.wait(1)

        mat_res = get_matrix([["160", "180"], ["180", "265"]])
        group_res = VGroup(eq_1.copy(), mat_res).arrange(RIGHT).shift(UP * 1.5)

        self.play(
            ReplacementTransform(group_calc[0], group_res[0]),
            ReplacementTransform(VGroup(group_calc[1], group_calc[2]), group_res[1]),
            run_time=1.5
        )
        self.wait(1)

        # --- 2. Hiển thị chữ "Diagonalization" ---
        title = Text("Diagonalization", font_size=40, color=WHITE).to_edge(UP)
        
        self.play(Write(title))
        self.play(group_res.animate.next_to(title, DOWN, buff=0.3).scale(0.8))
        self.wait(0.5)

        # --- 3. Trích xuất Trị riêng và Vector riêng ---
        tex_lambda = MathTex(
            r"\text{Eigenvalues: } \lambda_1 = 400, \quad \lambda_2 = 25", 
            color=YELLOW
        )
        
        tex_v = MathTex(
            r"\text{Eigenvectors: } v_1 = \frac{1}{5}\begin{bmatrix} 3 \\ 4 \end{bmatrix}, \quad v_2 = \frac{1}{5}\begin{bmatrix} -4 \\ 3 \end{bmatrix}", 
            color=GREEN
        )
        
        group_extract = VGroup(tex_lambda, tex_v).arrange(DOWN, buff=0.6).move_to(ORIGIN)

        self.play(FadeIn(tex_lambda, shift=DOWN))
        self.wait(0.5)
        self.play(FadeIn(tex_v, shift=DOWN))
        self.wait(1.5)

        # --- 4. Xếp thành công thức A^T A = V D V^-1 ---
        eq_diag = MathTex("A^T A", "=", "V", "D", "V^{-1}").scale(1.2)
        eq_diag[2].set_color(GREEN)
        eq_diag[3].set_color(YELLOW)
        eq_diag[4].set_color(GREEN)
        
        eq_diag.next_to(title, DOWN, buff=0.6) 

        self.play(
            FadeOut(group_res),
            ReplacementTransform(tex_lambda, eq_diag[3]), 
            ReplacementTransform(tex_v, eq_diag[2]),      
            Write(eq_diag[0:2]),
            Write(eq_diag[4]),
            run_time=2
        )
        self.wait(1)

        mat_V = VGroup(MathTex("V =", color=GREEN), MathTex(r"\frac{1}{5}", color=GREEN), get_matrix([["3", "-4"], ["4", "3"]], GREEN)).arrange(RIGHT, buff=0.1)
        mat_D = VGroup(MathTex("D =", color=YELLOW), get_matrix([["400", "0"], ["0", "25"]], YELLOW)).arrange(RIGHT, buff=0.1)
        mat_Vinv = VGroup(MathTex("V^{-1} =", color=GREEN), MathTex(r"\frac{1}{5}", color=GREEN), get_matrix([["3", "4"], ["-4", "3"]], GREEN)).arrange(RIGHT, buff=0.1)

        group_mats = VGroup(mat_V, mat_D, mat_Vinv).arrange(RIGHT, buff=0.6).next_to(eq_diag, DOWN, buff=0.5).scale(0.7)

        self.play(FadeIn(group_mats, shift=UP))
        self.wait(1.5)

        # --- 5. Suy ra Sigma và V^T ---
        arrow_D = Arrow(start=mat_D.get_bottom(), end=mat_D.get_bottom() + DOWN*0.5, color=YELLOW, buff=0.1)
        tex_sigma = VGroup(
            MathTex(r"\Sigma = \sqrt{D} =", color=YELLOW),
            get_matrix([["20", "0"], ["0", "5"]], YELLOW)
        ).arrange(RIGHT, buff=0.1).scale(0.7).next_to(arrow_D, DOWN, buff=0.1)
        
        arrow_V = Arrow(start=mat_Vinv.get_bottom(), end=mat_Vinv.get_bottom() + DOWN*0.5, color=GREEN, buff=0.1)
        tex_vt = VGroup(
            MathTex(r"V^T = V^{-1} =", color=GREEN),
            MathTex(r"\frac{1}{5}", color=GREEN),
            get_matrix([["3", "4"], ["-4", "3"]], GREEN)
        ).arrange(RIGHT, buff=0.1).scale(0.7).next_to(arrow_V, DOWN, buff=0.1)

        self.play(GrowArrow(arrow_D), GrowArrow(arrow_V))
        self.play(Write(tex_sigma), Write(tex_vt))
        self.wait(2)

        self.play(
            Indicate(tex_sigma, color=WHITE, scale_factor=1.1),
            Indicate(tex_vt, color=WHITE, scale_factor=1.1)
        )
        self.wait(2)

        self.play(*[FadeOut(mob) for mob in self.mobjects])
        self.wait(1)

    # ==========================================
    # Cảnh 9: Ứng dụng SVD nén ảnh (Minh hoạ ma trận)
    # ==========================================
    def scene_09_image_matrix_visualizer(self):
        # 1. Tải và điều chỉnh hình ảnh
        img = ImageMobject("image.jpg")
        img.scale_to_fit_height(config["frame_height"] * 0.5)
        img.to_edge(LEFT, buff=1.0)
        
        img_h = img.get_height()
        img_w = img.get_width()
        img_center = img.get_center()

        # 2. Tạo ma trận số ngẫu nhiên minh họa 
        num_cols = 20  
        num_rows = int(num_cols * (727 / 1024)) 

        matrix_data = np.random.randint(0, 256, size=(num_rows, num_cols))
        
        matrix = IntegerMatrix(
            matrix_data,
            include_background_rectangle=False,
            add_background_rectangles_to_entries=False,
            element_to_mobject_config={"font_size": 14} 
        )
        
        matrix.stretch_to_fit_height(img_h)
        matrix.stretch_to_fit_width(img_w)
        matrix.move_to(img_center)
        
        self.add(matrix)
        self.add(img)

        # --- Hoạt ảnh ---
        self.play(FadeIn(img), run_time=1.5)
        self.wait(1)

        self.play(
            matrix.animate.next_to(img, RIGHT, buff=0.8), 
            run_time=2.5,
            rate_func=smooth 
        )
        
        self.wait(2)
        self.play(*[FadeOut(mob) for mob in self.mobjects])

    # ==========================================
    # Cảnh 10: Minh hoạ nén ảnh bằng SVD và so sánh
    # ==========================================
    def scene_10_svd_compression(self):
        scale_factor_text = 0.9 
        side_length_pixel = 0.18 

        # --- 1. Ma trận ký hiệu và rút gọn công thức ---
        tex_A = MathTex("A", "=")
        tex_U = MathTex(
            r"\begin{bmatrix} | & | & | \\ ", 
            r"\vec{u}_1", r" & ", r"\vec{u}_2", r" & ", r"\vec{u}_3", 
            r" \\ | & | & | \end{bmatrix}"
        )
        tex_U[1].set_color(YELLOW); tex_U[3].set_color(YELLOW); tex_U[5].set_color(YELLOW)
        
        tex_Sig = MathTex(
            r"\begin{bmatrix} ", 
            r"\sigma_1", r" & 0 & 0 \\ 0 & ", 
            r"\sigma_2", r" & 0 \\ 0 & 0 & ", 
            r"\sigma_3", 
            r" \end{bmatrix}"
        )
        
        tex_VT = MathTex(
            r"\begin{bmatrix} - & ", 
            r"\vec{v}_1^T", r" & - \\ - & ", 
            r"\vec{v}_2^T", r" & - \\ - & ", 
            r"\vec{v}_3^T", 
            r" & - \end{bmatrix}"
        )
        tex_VT[1].set_color(BLUE); tex_VT[3].set_color(BLUE); tex_VT[5].set_color(BLUE)

        group_svd = VGroup(tex_A, tex_U, tex_Sig, tex_VT).arrange(RIGHT, buff=0.2)
        group_svd.shift(UP * 2.0).scale(scale_factor_text)

        self.play(Write(group_svd), run_time=1.5)
        self.wait(0.5)

        sum_expanded = MathTex(
            "A", "=", 
            r"\sigma_1", "(", r"\vec{u}_1", r"\vec{v}_1^T", ")", "+", 
            r"\sigma_2", "(", r"\vec{u}_2", r"\vec{v}_2^T", ")", "+", 
            r"\dots", "+",
            r"\sigma_r", "(", r"\vec{u}_r", r"\vec{v}_r^T", ")"
        )
        sum_expanded[2].set_color(WHITE); sum_expanded[4].set_color(YELLOW); sum_expanded[5].set_color(BLUE)
        sum_expanded[8].set_color(WHITE); sum_expanded[10].set_color(YELLOW); sum_expanded[11].set_color(BLUE)
        sum_expanded[16].set_color(WHITE); sum_expanded[18].set_color(YELLOW); sum_expanded[19].set_color(BLUE)
        
        sum_expanded.next_to(group_svd, DOWN, buff=1.2)

        self.play(
            Write(sum_expanded[0:2]),
            TransformFromCopy(tex_Sig[1], sum_expanded[2]),
            Write(sum_expanded[3]),
            TransformFromCopy(tex_U[1], sum_expanded[4]),
            TransformFromCopy(tex_VT[1], sum_expanded[5]),
            Write(sum_expanded[6]),
            run_time=1.5
        )
        self.play(
            Write(sum_expanded[7]),
            TransformFromCopy(tex_Sig[3], sum_expanded[8]),
            Write(sum_expanded[9]), 
            TransformFromCopy(tex_U[3], sum_expanded[10]),
            TransformFromCopy(tex_VT[3], sum_expanded[11]),
            Write(sum_expanded[12:]),
            run_time=1.5
        )
        self.wait(0.5)

        sum_general = MathTex(
            "A_k", "=", r"\sum_{i=1}^k", r"\sigma_i", r"\vec{u}_i", r"\vec{v}_i^T"
        )
        sum_general[3].set_color(WHITE); sum_general[4].set_color(YELLOW); sum_general[5].set_color(BLUE)
        sum_general.move_to(sum_expanded.get_center())

        self.play(
            ReplacementTransform(sum_expanded, sum_general),
            FadeOut(group_svd)
        )
        self.play(sum_general.animate.to_edge(UP, buff=0.3).scale(1.1))

        # --- 2. Minh hoạ nén ảnh 16x16 ---
        m, n = 16, 16
        A_img = np.zeros((m, n))
        for i in range(16):
            for j in range(16):
                if 3 <= i <= 12 and 3 <= j <= 12: 
                    if i in [3, 12] or j in [3, 12]: A_img[i, j] = 1.0 
                    elif i == j or i + j == 15: A_img[i, j] = 0.8    
                    else: A_img[i, j] = 0.2
                elif i in [1, 14] and 6 <= j <= 9: A_img[i, j] = 0.6 
                else: A_img[i, j] = 0.05

        U, S, VT = np.linalg.svd(A_img, full_matrices=False)

        def get_color(val):
            safe_val = float(np.clip(val, 0, 1)) 
            return rgb_to_color((safe_val, safe_val, safe_val))

        def get_grid(matrix):
            grid = VGroup()
            for r in range(m):
                for c in range(n):
                    square = Square(side_length=side_length_pixel)
                    square.set_fill(get_color(matrix[r, c]), opacity=1)
                    square.set_stroke(GRAY, width=0.2)
                    square.move_to(RIGHT * (c * side_length_pixel) + DOWN * (r * side_length_pixel))
                    grid.add(square)
            grid.center()
            return grid

        grid_orig = get_grid(A_img).shift(LEFT * 3.5 + DOWN * 0.5)
        label_orig = Text("Original Image", font_size=24).next_to(grid_orig, UP, buff=0.2)
        
        grid_comp = get_grid(np.zeros((m, n))).shift(RIGHT * 3.5 + DOWN * 0.5)
        label_comp = Text("Compressed (k=0)", font_size=24).next_to(grid_comp, UP, buff=0.2)

        self.play(FadeIn(grid_orig), Write(label_orig), FadeIn(grid_comp), Write(label_comp))

        # --- 3. Hiển thị dung lượng lưu trữ ---
        storage_orig_tex = Text(f"Storage: {m * n} data", font_size=20, color=RED).next_to(grid_orig, DOWN, buff=0.2)
        self.play(FadeIn(storage_orig_tex))

        storage_comp_tex = Text("Storage: 0 data", font_size=20, color=GREEN).next_to(grid_comp, DOWN, buff=0.2)
        self.play(FadeIn(storage_comp_tex))

        # --- 4. Hoạt ảnh tăng số k và so sánh ---
        k_tracker = ValueTracker(0)

        label_comp.add_updater(lambda mob: mob.become(
            Text(f"Compressed (k={int(round(k_tracker.get_value()))})", font_size=24).move_to(label_comp.get_center())
        ))

        def update_storage_text(mob):
            k = int(round(k_tracker.get_value()))
            storage_used = k * (m + n + 1)
            color = RED if storage_used >= (m * n) else GREEN
            mob.become(
                Text(f"Storage: {storage_used} data", font_size=20, color=color).move_to(storage_comp_tex.get_center())
            )

        storage_comp_tex.add_updater(update_storage_text)

        def update_pixel_colors(grid):
            k = int(round(k_tracker.get_value()))
            
            if k == 0:
                A_k = np.zeros((m, n))
            else:
                A_k = U[:, :k] @ np.diag(S[:k]) @ VT[:k, :]
            
            for i in range(m * n):
                r, c = divmod(i, n)
                grid[i].set_fill(get_color(A_k[r, c]), opacity=1)

        grid_comp.add_updater(update_pixel_colors)
        self.wait(1)
        
        self.play(k_tracker.animate.set_value(16), run_time=6, rate_func=linear)
        self.wait(1.5)

        self.play(k_tracker.animate.set_value(6), run_time=2.5)
        self.wait(2)

        label_comp.clear_updaters()
        storage_comp_tex.clear_updaters()
        grid_comp.clear_updaters()

        self.play(*[FadeOut(mob) for mob in self.mobjects])
        self.wait(1)
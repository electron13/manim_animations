import manim_ml
import manim
from manim import *
from manim_ml.neural_network import NeuralNetwork, Convolutional2DLayer, MaxPooling2DLayer, FeedForwardLayer, ImageLayer
from manim_ml.neural_network.animations.dropout import make_neural_network_dropout_animation

config.pixel_height = 700
config.pixel_width = 1900
config.frame_height = 12.0
config.frame_width = 12.0

class BasicScene(Scene):
    def construct(self):
        texx = Tex(r"a = b + c")
        self.play(Write(texx))
        

# Here we define our basic scene
class BasicScene1(Scene):
    def construct(self):
        text_intro = Text("This is a basic Feed Forward Neural Network", font_size=20)
        self.play(Write(text_intro), run_time=3)
        self.wait(0.5)
        self.remove(text_intro)
        nn = NeuralNetwork([
                FeedForwardLayer(3),
                FeedForwardLayer(7),
                FeedForwardLayer(7),
                FeedForwardLayer(2),
            ],
            layer_spacing=0.25,
        )
        nn.move_to(ORIGIN)
        self.add(nn)
        forward_pass = nn.make_forward_pass_animation()
        self.play(forward_pass, run_time=3)
        self.wait()
        nn.scale(0.3)
        self.add(nn.shift(DOWN))
        self.remove(nn)
        text_post_intro1 = Text("Soo..", font_size=20)
        self.play(Write(text_post_intro1), run_time=2)
        self.wait()
        self.remove(text_post_intro1)
        text_post_intro2 = Text("How exactly do Neural Networks work?", font_size=16)
        forward_text = Text('Neural Network Forward Pass', font_size=16)
        self.play(Write(text_post_intro2), run_time=4)
        self.wait(2)
        self.remove(text_post_intro2)

        input = MathTex(r'''\begin{bmatrix}
        i_1 \\
        i_2 \\
        \vdots \\
        i_n
    \end{bmatrix}''', color=YELLOW)
        inputtext = Text('Input', font_size=20)
        
        weight = MathTex(r'''\begin{bmatrix}
        a_{11} & a_{12} & a_{13} & a_{14} & a_{15} \\
        a_{21} & a_{22} & a_{23} & a_{24} & a_{25} \\
        \vdots & \vdots & \vdots & \vdots & \vdots \\
        a_{n1} & a_{n2} & a_{n3} & a_{n4} & a_{n5}
    \end{bmatrix}''', color=GREEN)
        weighttext = Text('Weights', font_size=20)
        
        bias = MathTex(r'''\begin{bmatrix}
        b_1 \\
        b_2 \\
        \vdots \\
        b_n
    \end{bmatrix}''', color=RED)
        biastext = Text('Biases', font_size=20)
        
        forward = MathTex(r'''
    \begin{bmatrix}
        x_1 \\
        x_2 \\
        \vdots \\
        x_n
    \end{bmatrix}
    =
    \begin{bmatrix}
        a_{11} & a_{12} & a_{13} & a_{14} & a_{15} \\
        a_{21} & a_{22} & a_{23} & a_{24} & a_{25} \\
        \vdots & \vdots & \vdots & \vdots & \vdots \\
        a_{n1} & a_{n2} & a_{n3} & a_{n4} & a_{n5}
    \end{bmatrix}
    \cdot
    \begin{bmatrix}
        i_1 \\
        i_2 \\
        \vdots \\
        i_n
    \end{bmatrix}
    +
    \begin{bmatrix}
        b_1 \\
        b_2 \\
        \vdots \\
        b_n
    \end{bmatrix}
        ''', color=BLUE)
        VGroup(forward_text,forward).arrange(DOWN, buff=0.05)
        VGroup(input,inputtext).arrange(DOWN, buff=0.05)
        VGroup(weight,weighttext).arrange(DOWN, buff=0.05)
        VGroup(bias,biastext).arrange(DOWN, buff=0.05)
        forward.set_stroke(BLACK, 5, background=True)
        forward.move_to(ORIGIN)
        forward.scale(0.4)
        input.scale(0.6)
        weight.scale(0.6)
        bias.scale(0.6)
        self.play(Write(forward))
        self.play(Write(forward_text))
        self.wait(3)
        self.remove(forward)
        self.remove(forward_text)
        self.play(Write(input))
        self.play(Write(inputtext))
        self.wait(1)
        self.remove(input)
        self.remove(inputtext)
        self.play(Write(weight))
        self.play(Write(weighttext))
        self.wait(1)
        self.remove(weight)
        self.remove(weighttext)
        self.play(Write(bias))
        self.play(Write(biastext))
        self.wait(1)
        self.remove(bias)
        self.remove(biastext)
        

class BasicScene2(Scene):
    def construct(self):
        sigmoid = MathTex(r''' 
    \sigma(x) = \frac{1}{1 + e^{-x}}
''', color=YELLOW)
        sigmoid.scale(0.5)
        #sigmoid_on_matrix = MathTex(r''' 
#\begin{bmatrix} X_1\\X_2\\\vdots\\X_n\end{bmatrix} = \sigma\left(\begin{bmatrix} x_1\\x_2\\\vdots\\x_n\end{bmatrix}\right)\;
#''')
        sigmoid_on_matrix = MathTex(r"\begin{bmatrix} X_1 \\ X_2 \\ \vdots \\ X_n \end{bmatrix} = \sigma\left( \begin{bmatrix} x_1 \\ x_2 \\ \vdots \\ x_n \end{bmatrix} \right)", color=BLUE)
        sigmoid_on_matrix.scale(0.5)
        activation_text = Text("Activation Functions", font_size=20)
        sigmoid_text = Text("Sigmoid Function", font="Consolas", font_size=14)
        VGroup(activation_text, sigmoid_text).arrange(DOWN, buff=1)
        self.play(Write(activation_text))
        self.play(FadeIn(sigmoid_text))
        self.wait(1)
        self.remove(activation_text)
        self.remove(sigmoid_text)
        self.play(Write(sigmoid))
        self.wait(1)
        self.remove(sigmoid)
        applying_sigmoid_text = Text("Applying the Sigmoid Function to the Neurons...", font_size=15)
        self.play(Write(applying_sigmoid_text))
        self.wait()
        self.remove(applying_sigmoid_text)
        self.play(Write(sigmoid_on_matrix))
        self.wait(3)
        self.remove(sigmoid_on_matrix)
        nn = NeuralNetwork([
                FeedForwardLayer(3),
                FeedForwardLayer(7,activation_function="Sigmoid"),
                FeedForwardLayer(7,activation_function="Sigmoid"),
                FeedForwardLayer(2,activation_function="Sigmoid"),
            ],
            layer_spacing=0.25,
        )
        
        nn.move_to(ORIGIN)
        self.add(nn)
        forward_pass = nn.make_forward_pass_animation()
        self.play(forward_pass)
        self.wait()
        

class BasicScene3(Scene):
  def construct(self):
    MSE = MathTex(r''' 
    
   L = (y_i - \hat{y_i})^2

''', color=YELLOW)
    loss_text = Text('Loss Functions', font_size=20)
    MSE_text = Text("Mean Square Error Loss Function", font="Consolas", font_size=16)
    VGroup(loss_text, MSE_text).arrange(DOWN, buff=1)
    self.play(Write(loss_text))
    self.play(FadeIn(MSE_text))
    self.wait(1)
    self.remove(loss_text)
    self.remove(MSE_text)
    MSE.scale(0.7)
    self.play(Write(MSE))
    self.wait(5)
    self.remove(MSE)
    applying_loss_text = Text('This finds deviation between the intended output and the Neural Network output', font_size=12)
    self.play(Write(applying_loss_text), run_time=4)
    self.wait(10)

class BasicScene4(Scene):
  def construct(self):
    objective = Text('Objective', font_size=20)
    objective_text = Text('To adjust the weights and biases of the Neural Network to approach minimum of loss', font="Consolas", font_size=9)
    VGroup(objective, objective_text).arrange(DOWN, buff=1)
    self.play(Write(objective))
    self.play(FadeIn(objective_text))
    self.wait()
    self.remove(objective)
    self.remove(objective_text)
    objective_text_2a = Text('For this, we need to know how much the change in each weight changes the loss, ', font_size=13)
    objective_text_2b = Text('the differential of the loss function with respect to the weight matrix', font_size = 13)
    dwdl = MathTex(r''' 
    \frac{\partial L}{\partial a}  ''' ,
    color=YELLOW
    )
    dwdl.scale(0.8)
    VGroup(objective_text_2a,objective_text_2b, dwdl).arrange(DOWN, buff=0.1)
    self.play(Write(objective_text_2a))
    self.play(Write(objective_text_2b))
    self.play(Write(dwdl))
    self.wait(3)
    self.remove(objective_text_2b)
    self.remove(objective_text_2a)
    self.remove(dwdl)
    objective_text_3 = Text('How do we get this?', font_size=20)
    self.play(Write(objective_text_3))
    self.remove(objective_text_3)
    loss_gradient = Text('Loss function derivative with respect to Output Layer Matrix', font_size=15)
    dldX = MathTex(r''' 
    \frac{\partial L}{\partial X} = 2(X_i - \hat{X_i})
''', color=ORANGE)
    VGroup(loss_gradient,dldX).arrange(DOWN, buff=1)
    self.play(Write(loss_gradient))
    self.play(Write(dldX))
    self.wait(4)
    self.remove(loss_gradient)
    self.remove(dldX)
    sigmoid_gradient = Text('Sigmoid function derivative with respect to Neuron Matrix', font_size=15)
    dXdx = MathTex(r''' 
    \frac{\partial X}{\partial x} = \sigma(x)\cdot(1-\sigma(x))
''', color=RED)
    VGroup(sigmoid_gradient,dXdx).arrange(DOWN, buff=1)
    self.play(Write(sigmoid_gradient))
    self.play(Write(dXdx))
    self.wait(4)
    self.remove(sigmoid_gradient)
    self.remove(dXdx)
    neuron_gradient = Text('Neuron Matrix derivative with respect to Weights', font_size=15)
    dxda = MathTex(r'''  
    \frac{\partial x}{\partial a} = \begin{bmatrix} \frac{\partial x_{11}}{\partial a} & \frac{\partial x_{12}}{\partial a} & \frac{\partial x_{13}}{\partial a} & \frac{\partial x_{14}}{\partial a} & \frac{\partial x_{15}}{\partial a}\\ \frac{\partial x_{21}}{\partial a} & \frac{\partial x_{22}}{\partial a} & \frac{\partial x_{23}}{\partial a} & \frac{\partial x_{24}}{\partial a} & \frac{\partial x_{25}}{\partial a} \\ \vdots & \vdots & \vdots & \vdots & \vdots \\ \frac{\partial x_{n1}}{\partial a} & \frac{\partial x_{n2}}{\partial a} & \frac{\partial x_{n3}}{\partial a} & \frac{\partial x_{n4}}{\partial a} & \frac{\partial x_{n5}}{\partial a} \end{bmatrix}''', color=BLUE)
    dxda.scale(0.4)
    VGroup(neuron_gradient,dxda).arrange(DOWN, buff=0.5)
    self.play(Write(neuron_gradient))
    self.play(Write(dxda))
    self.wait(4)
    self.remove(neuron_gradient)
    self.remove(dxda)
    chain_rule = Text('Using the Chain rule', font_size=15)
    final_eqn = MathTex(r''' 
    \frac{\partial L}{\partial a} = \frac{\partial L}{\partial X}\cdot\frac{\partial X}{\partial x}\cdot\frac{\partial x}{\partial a}
''', color=GREEN)
    VGroup(chain_rule,final_eqn).arrange(DOWN, buff=1)
    self.play(Write(chain_rule))
    self.play(Write(final_eqn))
    self.wait(7)
    self.remove(chain_rule)
    self.remove(final_eqn)
    nn = NeuralNetwork([
                FeedForwardLayer(3),
                FeedForwardLayer(7),
                FeedForwardLayer(7),
                FeedForwardLayer(2),
            ],
            layer_spacing=0.25,
        )
        
    nn.move_to(ORIGIN)
    self.add(nn)
    forward_pass = nn.make_forward_pass_animation()
    self.play(forward_pass)
    self.wait(10)

from PIL import Image


class BasicScene5(Scene):
  
  def construct(self):
    conv_networks = Text('Convolutional Neural Networks', font_size=20)
    self.play(Write(conv_networks))
    self.wait()
    self.remove(conv_networks)
    image = Image.open(r"C:\Users\Ashwin\Downloads\greyscale_prannay.png") 
    numpy_image = np.asarray(image)
    nn = NeuralNetwork([
            ImageLayer(numpy_image, height=1.5),
            Convolutional2DLayer(1, 7, 3, filter_spacing=0.32), # Note the default stride is 1. 
            Convolutional2DLayer(3, 5, 3, filter_spacing=0.32),
            Convolutional2DLayer(5, 3, 3, filter_spacing=0.18),
            FeedForwardLayer(3),
            FeedForwardLayer(3),
        ],
        layer_spacing=0.25,
    )
    
    nn.move_to(ORIGIN)
    self.add(nn)
    forward_pass = nn.make_forward_pass_animation()
    self.play(forward_pass)
    self.wait(5)

class BasicScene6(ThreeDScene):
  def construct(self):
    dropout = Text('Dropout', font_size=20)
    self.play(Write(dropout))
    self.wait()
    self.remove(dropout)
    nn = NeuralNetwork([
        FeedForwardLayer(3),
        FeedForwardLayer(5),
        FeedForwardLayer(3),
        FeedForwardLayer(5),
        FeedForwardLayer(4),
    ],
    layer_spacing=0.4,
    )   
    nn.move_to(ORIGIN)
    self.add(nn)
    self.play(
        make_neural_network_dropout_animation(
            nn, dropout_rate=0.25, do_forward_pass=True
        )
    )
    self.wait(10)




# Warning, this file uses ContinualChangingDecimal,
# which has since been been deprecated.  Use a mobject
# updater instead


class GradientDescentWrapper(Scene):
    def construct(self):
        title = Tex("Gradient descent")
        title.to_edge(UP)
        rect = ScreenRectangle(height=6)
        rect.next_to(title, DOWN)

        self.add(title)
        self.play(ShowCreation(rect))
        self.wait()


class ShowSimpleMultivariableFunction(Scene):
    def construct(self):
        scale_val = 1.5

        func_tex = Tex(
            "C(", "x_1,", "x_2,", "\\dots,", "x_n", ")", "=",
        )
        func_tex.scale(scale_val)
        func_tex.shift(2 * LEFT)
        alt_func_tex = Tex(
            "C(", "x,", "y", ")", "="
        )
        alt_func_tex.scale(scale_val)
        for tex in func_tex, alt_func_tex:
            tex.set_color_by_tex_to_color_map({
                "C(": RED,
                ")": RED,
            })
        alt_func_tex.move_to(func_tex, RIGHT)
        inputs = func_tex[1:-2]
        self.add(func_tex)

        many_inputs = Tex(*[
            "x_{%d}, " % d for d in range(1, 25)
        ])
        many_inputs.set_width(FRAME_WIDTH)
        many_inputs.to_edge(UL)

        inputs_brace = Brace(inputs, UP)
        inputs_brace_text = inputs_brace.get_text("Multiple inputs")

        decimal = DecimalNumber(0)
        decimal.scale(scale_val)
        decimal.next_to(tex, RIGHT)
        value_tracker = ValueTracker(0)
        always_shift(value_tracker, rate=0.5)
        self.add(value_tracker)
        decimal_change = ContinualChangingDecimal(
            decimal,
            lambda a: 1 + np.sin(value_tracker.get_value())
        )
        self.add(decimal_change)

        output_brace = Brace(decimal, DOWN)
        output_brace_text = output_brace.get_text("Single output")

        self.wait(2)
        self.play(GrowFromCenter(inputs_brace))
        self.play(Write(inputs_brace_text))
        self.play(GrowFromCenter(output_brace))
        self.play(Write(output_brace_text))
        self.wait(3)
        self.play(
            ReplacementTransform(
                inputs,
                many_inputs[:len(inputs)]
            ),
            LaggedStartMap(
                FadeIn,
                many_inputs[len(inputs):]
            ),
            FadeOut(inputs_brace),
            FadeOut(inputs_brace_text),
        )
        self.wait()
        self.play(
            ReplacementTransform(
                func_tex[0], alt_func_tex[0]
            ),
            Write(alt_func_tex[1:3]),
            LaggedStartMap(FadeOutAndShiftDown, many_inputs)
        )
        self.wait(3)


#class ShowGraphWithVectors(ExternallyAnimatedScene):
    #pass


class ShowFunction(Scene):
    def construct(self):
        func = Tex(
            "f(x, y) = e^{-x^2 + \\cos(2y)}",
            tex_to_color_map={
                "x": BLUE,
                "y": RED,
            }
        )
        func.scale(1.5)
        self.play(FadeIn(func))
        self.wait()


#class ShowExampleFunctionGraph(ExternallyAnimatedScene):
    #pass


class ShowGradient(Scene):
    def construct(self):
        lhs = MathTex(
            "\\nabla f(x, y)=",
            tex_to_color_map={"x": BLUE, "y": RED}
        )
        vector = Matrix([
            ["\\partial f / \\partial x"],
            ["\\partial f / \\partial y"],
        ], v_buff=1)
        gradient = VGroup(lhs, vector)
        gradient.arrange(RIGHT, buff=SMALL_BUFF)
        gradient.scale(1.5)

        del_x, del_y = partials = vector.get_entries()
        background_rects = VGroup()
        for partial, color in zip(partials, [BLUE, RED]):
            partial[-1].set_color(color)
            partial.rect = SurroundingRectangle(
                partial, buff=MED_SMALL_BUFF
            )
            partial.rect.set_stroke(width=0)
            partial.rect.set_fill(color=color, opacity=0.5)
            background_rects.add(partial.rect.copy())
        background_rects.set_fill(opacity=0.1)

        partials.set_fill(opacity=0)

        self.play(
            LaggedStartMap(FadeIn, gradient),
            LaggedStartMap(
                FadeIn, background_rects,
                rate_func=squish_rate_func(smooth, 0.5, 1)
            )
        )
        self.wait()
        for partial in partials:
            self.play(DrawBorderThenFill(partial.rect))
            self.wait()
            self.play(FadeOut(partial.rect))
        self.wait()
        for partial in partials:
            self.play(Write(partial))
            self.wait()


#class ExampleGraphHoldXConstant(ExternallyAnimatedScene):
    #pass


#class ExampleGraphHoldYConstant(ExternallyAnimatedScene):
    #pass


class TakePartialDerivatives(Scene):
    def construct(self):
        tex_to_color_map = {
            "x": BLUE,
            "y": RED,
        }
        func_tex = MathTex(
            "f", "(", "x", ",", "y", ")", "=",
            "e^{", "-x^2", "+ \\cos(2y)}",
            tex_to_color_map=tex_to_color_map
        )
        partial_x = MathTex(
            "{\\partial", "f", "\\over", "\\partial", "x}", "=",
            "\\left(", "e^", "{-x^2", "+ \\cos(2y)}", "\\right)",
            "(", "-2", "x", ")",
            tex_to_color_map=tex_to_color_map,
        )
        partial_y = MathTex(
            "{\\partial", "f", "\\over", "\\partial", "y}", "=",
            "\\left(", "e^", "{-x^2", "+ \\cos(", "2", "y)}", "\\right)",
            "(", "-\\sin(", "2", "y)", "\\cdot 2", ")",
            tex_to_color_map=tex_to_color_map,
        )
        partials = VGroup(partial_x, partial_y)
        for mob in func_tex, partials:
            mob.scale(1.5)

        func_tex.move_to(2 * UP + 3 * LEFT)
        for partial in partials:
            partial.next_to(func_tex, DOWN, buff=LARGE_BUFF)
            top_eq_x = func_tex.get_part_by_tex("=").get_center()[0]
            low_eq_x = partial.get_part_by_tex("=").get_center()[0]
            partial.shift((top_eq_x - low_eq_x) * RIGHT)

        index = func_tex.index_of_part_by_tex("e^")
        exp_rect = SurroundingRectangle(func_tex[index + 1:], buff=0)
        exp_rect.set_stroke(width=0)
        exp_rect.set_fill(GREEN, opacity=0.5)

        xs = func_tex.get_parts_by_tex("x", substring=False)
        ys = func_tex.get_parts_by_tex("y", substring=False)
        for terms in xs, ys:
            terms.rects = VGroup(*[
                SurroundingRectangle(term, buff=0.5 * SMALL_BUFF)
                for term in terms
            ])
            terms.arrows = VGroup(*[
                Vector(0.5 * DOWN).next_to(rect, UP, SMALL_BUFF)
                for rect in terms.rects
            ])
        treat_as_constant = Tex("Treat as a constant")
        treat_as_constant.next_to(ys.arrows[1], UP)

        # Start to show partial_x
        self.play(FadeIn(func_tex))
        self.wait()
        self.play(
            ReplacementTransform(func_tex[0].copy(), partial_x[1]),
            Write(partial_x[0]),
            Write(partial_x[2:4]),
            Write(partial_x[6]),
        )
        self.play(
            ReplacementTransform(func_tex[2].copy(), partial_x[4])
        )
        self.wait()

        # Label y as constant
        #self.play(LaggedStartMap(ShowCreation, ys.rects))
        self.play(
            LaggedStartMap(GrowArrow, ys.arrows, lag_ratio=0.8),
            Write(treat_as_constant)
        )
        self.wait(2)

        # Perform partial_x derivative
        self.play(FadeIn(exp_rect), Animation(func_tex))
        self.wait()
        pxi1 = 8
        pxi2 = 15
        self.play(
            ReplacementTransform(
                func_tex[7:].copy(),
                partial_x[pxi1:pxi2],
            ),
            FadeIn(partial_x[pxi1 - 1:pxi1]),
            FadeIn(partial_x[pxi2]),
        )
        self.wait(2)
        self.play(
            ReplacementTransform(
                partial_x[10:12].copy(),
                partial_x[pxi2 + 2:pxi2 + 4],
                path_arc=-(TAU / 4)
            ),
            FadeIn(partial_x[pxi2 + 1]),
            FadeIn(partial_x[-1]),
        )
        self.wait(2)

        # Swap out partial_x for partial_y
        self.play(
            FadeOut(partial_x),
            FadeOut(ys.rects),
            FadeOut(ys.arrows),
            FadeOut(treat_as_constant),
            FadeOut(exp_rect),
            Animation(func_tex)
        )
        self.play(FadeIn(partial_y[:7]))
        self.wait()

        treat_as_constant.next_to(xs.arrows[1], UP, SMALL_BUFF)
        self.play(
            #LaggedStartMap(ShowCreation, xs.rects),
            LaggedStartMap(GrowArrow, xs.arrows),
            Write(treat_as_constant),
            lag_ratio=0.8
        )
        self.wait()

        # Show same outer derivative
        self.play(
            ReplacementTransform(
                func_tex[7:].copy(),
                partial_x[pxi1:pxi2],
            ),
            FadeIn(partial_x[pxi1 - 2:pxi1]),
            FadeIn(partial_x[pxi2]),
        )
        self.wait()
        self.play(
            ReplacementTransform(
                partial_y[12:16].copy(),
                partial_y[pxi2 + 3:pxi2 + 7],
                path_arc=-(TAU / 4)
            ),
            FadeIn(partial_y[pxi2 + 2]),
            FadeIn(partial_y[-1]),
        )
        self.wait()
        self.play(ReplacementTransform(
            partial_y[-5].copy(),
            partial_y[-2],
            path_arc=-PI
        ))
        self.wait()


class ShowDerivativeAtExamplePoint(Scene):
    def construct(self):
        tex_to_color_map = {
            "x": BLUE,
            "y": RED,
        }
        func_tex = Tex(
            "f", "(", "x", ",", "y", ")", "=",
            "e^{", "-x^2", "+ \\cos(2y)}",
            tex_to_color_map=tex_to_color_map
        )
        gradient_tex = Tex(
            "\\nabla", "f", "(", "x", ",", "y", ")", "=",
            tex_to_color_map=tex_to_color_map
        )

        partial_vect = Matrix([
            ["{\\partial f / \\partial x}"],
            ["{\\partial f / \\partial y}"],
        ])
        partial_vect.get_mob_matrix()[0, 0][-1].set_color(BLUE)
        partial_vect.get_mob_matrix()[1, 0][-1].set_color(RED)
        result_vector = self.get_result_vector("x", "y")

        gradient = VGroup(
            gradient_tex,
            partial_vect,
            Tex("="),
            result_vector
        )
        gradient.arrange(RIGHT, buff=SMALL_BUFF)

        func_tex.to_edge(UP)
        gradient.next_to(func_tex, DOWN, buff=LARGE_BUFF)

        example_lhs = Tex(
            "\\nabla", "f", "(", "1", ",", "3", ")", "=",
            tex_to_color_map={"1": BLUE, "3": RED},
        )
        example_result_vector = self.get_result_vector("1", "3")
        example_rhs = DecimalMatrix([[-1.92], [0.54]])
        example = VGroup(
            example_lhs,
            example_result_vector,
            Tex("="),
            example_rhs,
        )
        example.arrange(RIGHT, buff=SMALL_BUFF)
        example.next_to(gradient, DOWN, LARGE_BUFF)

        self.add(func_tex, gradient)
        self.wait()
        self.play(
            ReplacementTransform(gradient_tex.copy(), example_lhs),
            ReplacementTransform(result_vector.copy(), example_result_vector),
        )
        self.wait()
        self.play(Write(example[2:]))
        self.wait()

    def get_result_vector(self, x, y):
        result_vector = Matrix([
            ["e^{-%s^2 + \\cos(2\\cdot %s)} (-2\\cdot %s)" % (x, y, x)],
            ["e^{-%s^2 + \\cos(2\\cdot %s)} \\big(-\\sin(2\\cdot %s) \\cdot 2\\big)" % (x, y, y)],
        ], v_buff=1.2, element_alignment_corner=ORIGIN)

        x_terms = VGroup(
            result_vector.get_mob_matrix()[0, 0][2],
            result_vector.get_mob_matrix()[1, 0][2],
            result_vector.get_mob_matrix()[0, 0][-2],
        )
        y_terms = VGroup(
            result_vector.get_mob_matrix()[0, 0][11],
            result_vector.get_mob_matrix()[1, 0][11],
            result_vector.get_mob_matrix()[1, 0][-5],
        )
        x_terms.set_color(BLUE)
        y_terms.set_color(RED)
        return result_vector
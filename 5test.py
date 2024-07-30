class A():
    def __init__(self, init_age):
        super().__init__()
        print('我年龄是:',init_age)
        self.age = init_age
 
    def __call__(self, added_age):
        
 
        res = self.forward(added_age)
        return res
 
    def forward(self, input_):
        print('forward 函数被调用了')
        
        return input_ + self.age
print('对象初始化。。。。')
a = A(10)
 
 
input_param = a(2)
print("我现在的年龄是：", input_param)
 
 
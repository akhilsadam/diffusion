# projection / evaluation modifiers...abs

def DPS(denoise, A, y, gamma):
    
    def DPS_denoise(x):
        xd = denoise(x)
        
        with torch.enable_grad():
            x = x.requires_grad_(True)

            y_hat = A(x)
            err = F.mse_loss(y_hat, y)

            g = torch.autograd.grad(outputs=err,inputs=x,grad_outputs=torch.ones_like(err))[0] 

        x = xd - 0.5 * gamma * g # DPS update
        
        return x

    # usage
    # fmm = FMM()
    # fmm.denoise = DPS(fmm.denoise, A, y, gamma)
    # generate with # fmm.gen(z)

    

    
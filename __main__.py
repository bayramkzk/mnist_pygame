from model_manager import MnistModelManager
import pygame
import numpy as np

pygame.init()

# ------------------------------ CONSTANTS ------------------------------------

CAPTION = 'Digit Recognition'
SIZE = 560
FPS = 60

PIXEL_SIZE = SIZE // 28

assert SIZE % 28 == 0, 'size must be divisible by 28'
assert isinstance(FPS, int), 'fps must be an int'

DISPLAY_FONT = pygame.font.SysFont('monospace', 25)

# -------------------- PREDICT DIGIT ACCORDING TO ARRAY -----------------------

def predict(model, arr):
    x = np.array([arr])
    p_arr = model.predict(x)[0]
    p = max(range(len(p_arr)), key=p_arr.__getitem__)
    return p

# --------------------------- DRAW ARRAY TO SURFACE ---------------------------

def draw_arr(surface, arr):
    for y_index in range(28):
        for x_index in range(28):
            c = arr[y_index, x_index] * 255
            rect = (x_index * PIXEL_SIZE, y_index * PIXEL_SIZE, 
                    PIXEL_SIZE, PIXEL_SIZE)
            pygame.draw.rect(surface, (c, c, c), rect)

# ------------------------------- MAIN FUNCTION -------------------------------

def main():
    mgr = MnistModelManager()
    # the model can be refitted with MnistModelManager.fit_model() and
    # can be saved with MnistModelManager.save_model()
    mgr.read_model('mnist_model.json', 'mnist_model.h5')    
    
    screen = pygame.display.set_mode([SIZE, SIZE])
    pygame.display.set_caption(CAPTION)
    
    screen_arr = np.zeros([28, 28])
    last_mouse_pressed = False
    
    draw_arr(screen, screen_arr)
    
    clock = pygame.time.Clock()
    done = False
    
    while not done:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                done = True
    
        # --------------------------- MOUSE OPERATIONS ------------------------
        
        mouse_x, mouse_y = pygame.mouse.get_pos()
        mouse_left_pressed, _, mouse_right_pressed = pygame.mouse.get_pressed()
        
        if mouse_left_pressed:
            x_index = mouse_x // PIXEL_SIZE
            y_index = mouse_y // PIXEL_SIZE
            
            screen_arr[y_index, x_index] += 0.5
            
            if screen_arr[y_index, x_index] > 1:
                screen_arr[y_index, x_index] = 1
            
            draw_arr(screen, screen_arr)
            
            # set last_mouse_pressed to True for next loop
            last_mouse_pressed = True
        
        # if mouse was pressed in the previus loop but not now 
        elif last_mouse_pressed:
            # predict digit according to 28x28 array
            digit = predict(mgr.model, screen_arr)
            
            # displaying digit
            digit_render = DISPLAY_FONT.render(str(digit), True, (255, 255, 255))
            
            # display predicted digit in top left corner
            screen.blit(digit_render, (10, 10))
            
            # set mouse pressed to False for next loop, so a value 
            # won't be predicted in the next loop unnecessarily
            last_mouse_pressed = False
        
        elif mouse_right_pressed:
            screen_arr = np.zeros([28, 28])
            draw_arr(screen, screen_arr)
        
        # ------------------- REFRESH WINDOW AND SYNC FPS ---------------------        
        
        pygame.display.flip()
        
        clock.tick(FPS)

# ---------------------- RUN MAIN IF FILE WAS NOT IMPORTED --------------------

if __name__ == '__main__':
    main()

import pygame
import time
import random

pygame.init()

#colors
yellow = (255, 255, 102)
green = (0,255,0)
blue = (50,153,213)
red = (255,0,0)
white = (255,255,255)
black = (0,0,0)


#display
dis_width = 400
dis_height = 300

dis = pygame.display.set_mode((dis_width,dis_height))
pygame.display.set_caption('Snake')

clock = pygame.time.Clock()
#game

snake_block = 10
snake_speed = 30

font_style = pygame.font.SysFont("comicsansms", 12)
score_font = pygame.font.SysFont("comicsansms", 17)
def Your_score(score):
    value = score_font.render("Your Score: " + str(score), True, yellow)
    dis.blit(value, [0,0])
    
def our_snake(snake_block, snake_list):
    for x in snake_list:
        pygame.draw.rect(dis,black, [x[0], x[1], snake_block, snake_block])

def message(msg, color):
    mesg = font_style.render(msg, True, color)
    dis.blit(mesg, [dis_width/6, dis_height/3])

def gameLoop():
    game_over = False
    game_close = False
    
    x1 = dis_width/2
    y1 = dis_height/2
    
    x1delta = 0
    y1delta = 0

    snake_List = []
    Length_of_snake = 1
    
    foodx1 = round(random.randrange(0, dis_width - snake_block - 35)/10.0) * 10.0
    foody1 = round(random.randrange(0, dis_height - snake_block - 35)/10.0) * 10.0
    foodx2 = round(random.randrange(0, dis_width - snake_block - 35)/10.0) * 10.0
    foody2 = round(random.randrange(0, dis_height - snake_block - 35)/10.0) * 10.0
    

    while not game_over:

        while game_close == True:
            dis.fill(blue)
            message("You Lost! Press Q-Quit or C-Play Again", red)
            Your_score(Length_of_snake - 1)
            pygame.display.update()
            
            for event in pygame.event.get():
                if event.type == (pygame.KEYDOWN):
                    if event.key == pygame.K_q:
                        game_over = True
                        game_close = False
                    if event.key == pygame.K_c:
                        gameLoop()
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                game_over = True
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_LEFT or event.key == pygame.K_a:
                    x1delta = -snake_block
                    y1delta = 0
                elif event.key == pygame.K_RIGHT or event.key == pygame.K_d:
                    x1delta = snake_block
                    y1delta = 0
                elif event.key == pygame.K_UP or event.key == pygame.K_w:
                    x1delta = 0
                    y1delta = -snake_block
                elif event.key == pygame.K_DOWN or event.key == pygame.K_s:
                    x1delta = 0
                    y1delta = snake_block
        if x1 >= dis_width or x1 < 0 or y1 >= dis_height or y1 < 0:
            game_close = True
        x1 += x1delta
        y1 += y1delta
        pygame.display.update()
        dis.fill(blue)
        pygame.draw.rect(dis, green, [foodx1, foody1, snake_block, snake_block])
        pygame.draw.rect(dis, green, [foodx2, foody2, snake_block, snake_block])
        pygame.display.update()
        snake_Head = []
        snake_Head.append(x1)
        snake_Head.append(y1)
        snake_List.append(snake_Head)
        if len(snake_List) > Length_of_snake:
            del snake_List[0]

        for x in snake_List[:-1]:
            if x == snake_Head:
                game_close = True

        our_snake(snake_block, snake_List)
        Your_score(Length_of_snake - 1)
        
        pygame.display.update()
        
        if (x1 == foodx1 and y1 == foody1):
            foodx1 = round(random.randrange(0, dis_width - snake_block - 35)/10.0) * 10.0
            foody1 = round(random.randrange(0, dis_height - snake_block - 35)/10.0) * 10.0
            Length_of_snake += 1

            
        if ((x1 == foodx2 and y1 == foody2)):
            foodx2 = round(random.randrange(0, dis_width - snake_block - 35)/10.0) * 10.0
            foody2 = round(random.randrange(0, dis_height - snake_block - 35)/10.0) * 10.0
            Length_of_snake += 1

        pygame.display.update()
            
        clock.tick(snake_speed)


    pygame.quit()
    quit()
gameLoop()


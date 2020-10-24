<p align="center">
  <a href="https://github.com/lRomul/gramtion"><img src="https://raw.githubusercontent.com/lRomul/gramtion/master/pics/gramtion.jpeg" alt="Title GRAMTION on the background of a black and white photo of clouds in the sky"></a>
</p>
<p align="center">
    <em>Twitter bot for generating photo descriptions (alt text)</em>
</p>

This repo contains the source code of the Twitter [@GramtionBot](https://twitter.com/GramtionBot) for generating photo descriptions.  
Use cases and intends: 
* Help visually impaired Twitter users. 
Good image descriptions will help them understand what is happening in an image. 
Instagram and Facebook use deep learning for image captioning. 
Twitter users can only add custom alt text descriptions themselves. 
Automation of alt text generation will help Twitter be more accessible. 
* Collect dataset for image captioning (if it's legal, I don't know now). 
Annotations can be done by creating polls about prediction quality and getting corrected descriptions from users. 
Twitter API v1.1 has not the ability to create polls, but it will be added to API v2. 

## How to use

Tweet photo with mention [@GramtionBot](https://twitter.com/GramtionBot) or reply with mention to a tweet with a photo and the bot will send you an auto-generated image description.

![example 1](https://raw.githubusercontent.com/lRomul/gramtion/master/pics/example1.jpeg) ![example 2](https://raw.githubusercontent.com/lRomul/gramtion/master/pics/example2.jpeg)

Links to [example 1](https://twitter.com/GramtionBot/status/1318674709874118656) and [example 2](https://twitter.com/snoWhite_tan/status/963953292383580165) tweets.

## Dependencies
Gramtion is mostly made from ready parts:
* Model taken from [self-critical.pytorch](https://github.com/ruotianluo/self-critical.pytorch)
* Bot written with [Tweepy](https://github.com/tweepy/tweepy)
* Configuration settings implemented with [pydantic](https://github.com/samuelcolvin/pydantic/).

## Run own bot

To run your instance of the bot you need to install [Docker](https://www.docker.com/) and create [Twitter API auth credentials](https://realpython.com/twitter-bot-python-tweepy/#creating-twitter-api-authentication-credentials).  
If you have a Twitter developer account, but don't want to use it as a bot username, you can authenticate a new user that’s not has a developer account with [twurl](https://github.com/twitter/twurl).

* Create .env file with credentials. 

    ```
    CONSUMER_KEY={{ consumer_key }}
    CONSUMER_SECRET={{ consumer_secret }}
    ACCESS_TOKEN={{ access_token }}
    ACCESS_TOKEN_SECRET={{ access_token_secret }}
    ```

* Run Docker container with running the bot  

    ```bash
    docker run -d --restart=always --gpus=all --env-file .env --name=gramtion ghcr.io/lromul/gramtion:0.0.2
    ```

* Open bot logs 

    ```bash
    docker logs -f gramtion
    ```

* Stop bot

    ```bash
    docker stop gramtion
    docker rm gramtion
    ```

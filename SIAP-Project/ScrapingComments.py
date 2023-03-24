import pandas as pd
import praw

def ScrapeComments(dataset, startIndex):
    print('REDDIT comments')

    reddit = praw.Reddit(
    client_id='dh8WbMKCMJGcuT5CoBNMPg',
    client_secret='xX058y-B9Dc738svgIX5ni40kgkvtA',
    user_agent='python:com.example.myredditapp:v1.0.0 (by /u/nnatasas)',
    )
    
    subreddit = reddit.subreddit('television')

    # List of TV show keywords to search for
    tv_shows = dataset['Title']

    # Create a list to hold the data
    data = []

    iterator = 0
    # Search for submissions containing the TV show keywords
    for tv_show in tv_shows:
        iterator = iterator + 1
        if iterator < startIndex:
            continue
        if iterator >= startIndex + 100:
            break

        print(tv_show)
        submissions = subreddit.search(tv_show, sort='relevance', limit=100)
        comment_count = 0
    
        # Iterate over the submissions and their comments
        for submission in submissions:
            submission.comments.replace_more(limit=None)
            for comment in submission.comments.list():
                # Append a dictionary with the TV show name and comment body to the data list
                data.append({'TV Show': tv_show, 'Comment': comment.body})
            
                comment_count += 1
                if comment_count == 100:
                    break
        
            if comment_count == 100:
                break

    # Create a pandas DataFrame from the data list
    df = pd.DataFrame(data)

    # Write the DataFrame to a CSV file
    df.to_csv(f'reddit_comments{startIndex}.csv', index=False, encoding='utf-8')

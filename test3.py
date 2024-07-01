from dotenv import load_dotenv
import os
load_dotenv()
groq_api_key = os.environ['GROQ_API_KEY']

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_groq import ChatGroq
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain.tools.retriever import create_retriever_tool
from langchain.tools import Tool
from langchain_core.messages import HumanMessage, AIMessage
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_community.embeddings import OllamaEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import Chroma

import threading

import cv2
import argparse
from ultralytics import YOLO
import time
import requests
import concurrent.futures
headers = {"Authorization": "Bearer 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"}
vision = []

def parse_arguments():
    parser = argparse.ArgumentParser(description="YOLOv8 live")
    parser.add_argument(
        "--webcam-resolution",
        default=[1280, 720],
        nargs=2,
        type=int
        )
    args = parser.parse_args()
    return args

def query(frame, API_URL):
    _, img_encoded = cv2.imencode('.jpg', frame)
    data = img_encoded.tobytes()
    response = requests.post(API_URL, headers=headers, data=data)
    return response.json()

# vision = ""


def main1():
    # vision = []
    # nonlocal vision
    global vision
    args = parse_arguments()
    frame_width, frame_height = args.webcam_resolution
    
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, frame_width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, frame_height)
    
    model = YOLO("yolov8l.pt", verbose=False)
    limit = 0
    vision = ""
    while True:
        res, frame = cap.read()
        if res:
            result = model(frame)[0]
            
            try:
                
                try:
                    with concurrent.futures.ThreadPoolExecutor() as executor:
                        future1 = executor.submit(query, frame, "https://api-inference.huggingface.co/models/moranyanuka/blip-image-captioning-large-mocha")
                        future2 = executor.submit(query, frame, "https://api-inference.huggingface.co/models/Salesforce/blip-image-captioning-large")
                        
                        caption1 = future1.result()
                        caption2 = future2.result()
                        
                    caption = "Observation 1 :" + caption1[0]['generated_text'] + " Observation 2 : " + caption2[0]['generated_text']
                except:
                    caption = ""
                    
                # print("Caption: ", caption)
                
                object_counts = {}
                
                for box in result.boxes:
                    class_id = result.names[box.cls[0].item()]
                    object_counts[class_id] = object_counts.get(class_id, 0) + 1
                    
                # print(object_counts)
                oc = ""
                for i in object_counts:
                    oc += "Number of " + i + " is " +str(object_counts[i])
                    oc += " "
                # print(oc)
                
                # res = []
                # for box in result.boxes:
                #     class_id = result.names[box.cls[0].item()]
                #     cords = box.xyxy[0].tolist()
                #     cords = [round(x) for x in cords]
                #     conf = round(box.conf[0].item(), 2)
                #     res.append({
                #         "Object type": class_id,
                #         "Coordinates": cords,
                #         "Probability": conf,
                #     })
                # ans = [caption, oc, res]
                vision += caption + " " + oc
                vision += " "
                limit += 1
                if limit == 5:
                    vision = ""
                    vision += caption + " " + oc
                    limit = 0
                # print("Vision in Function = ", vision)
            except Exception as e:
                
                print("Not Working: ", e)
                continue
            
            if (cv2.waitKey(0) == 27):
                break
            
            
            time.sleep(5)
            
            
    # cap.release()
    # cv2.destroyAllWindows()

# Model Starts from Here !

model = ChatGroq(
    groq_api_key = groq_api_key,
    model_name = "llama3-70b-8192",
    temperature=0.5
)

# loader = WebBaseLoader("https://www.healthline.com/health/vaginal-discharge-color-guide")
# docs = loader.load()

# splitter = RecursiveCharacterTextSplitter(
#     chunk_size = 200,
#     chunk_overlap = 20
# )

# splitDocs = splitter.split_documents(docs)
# embeddings = OllamaEmbeddings(model='nomic-embed-text')
# vectorStore = Chroma.from_documents(splitDocs, embedding=embeddings)
# retriever = vectorStore.as_retriever(search_kwargs={"k":2})


# prompt = ChatPromptTemplate.from_messages([
#     ("system", 
#         ''' 
#         Imagine being a friendly AI assistant with visual perception.
#         Engage in natural conversations, starting with observations about your surroundings.
#         Make jokes and be warm, friendly, and engaging like a human friend. Respond to user requests and adapt to their tone and language.
#         Use your visual perception to inform responses and make them relatable. Be creative, but also consider the user's tone and language.
#         Assist the user in the best way possible, using tools when needed.
#         Don't hallucinate the vision - if you can't see anything, just say so.
#         '''
#                 ),
    
#     MessagesPlaceholder(variable_name="chat_history"),
#     ("human", "{input}"),
#     MessagesPlaceholder(variable_name="agent_scratchpad")
# ])


# print("Vision in prompt = ", vision)

# search = TavilySearchResults()

# retriver_tool = create_retriever_tool(
#     retriever,
#     "Menstruation",
#     # "Use this tool when searching for information about menstrual health"
#     "Dont use this tool. This is for testing purposes only"
# )

# tools=[search, retriver_tool]
# tools=[search]

# agent = create_tool_calling_agent(
#     llm = model,
#     prompt=prompt,
#     tools=tools
# )

# agentExecutor = AgentExecutor(
#     agent=agent,
#     tools=tools,
#     verbose=True
# )





def main2():
    
    from dotenv import load_dotenv
    import os
    load_dotenv()
    groq_api_key = os.environ['GROQ_API_KEY']
    global vision
    chat_history = []
    # global prompt
    model = ChatGroq(
    groq_api_key = groq_api_key,
    model_name = "llama3-70b-8192",
    temperature=0.5
    )
    
    def process_chat(agentExecutor, user_input, chat_history):
        
        print('processing...')
        
        response = agentExecutor.invoke({
            "input": user_input,
            "chat_history": chat_history
        })
        # print("Sent")
        return response['output']

    while True:
        user_input = input("You: ")
        prompt = ChatPromptTemplate.from_messages([
        ("system", 
            ''' 
            Imagine being a friendly AI assistant with visual perception.
            Engage in natural conversations, starting with observations about your surroundings.
            Make jokes and be warm, friendly, and engaging like a human friend. Respond to user requests and adapt to their tone and language.
            Use your visual perception to inform responses and make them relatable. Be creative, but also consider the user's tone and language.
            Assist the user in the best way possible, using tools when needed.
            Don't hallucinate the vision - if you can't see anything, just say so.
            Vision Inputs: {vision}
            '''
                    ),
        
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad")
        ])
        
        # vision = ""
        # print(vision)
        prompt = prompt.partial(vision=vision)
        # prompt = prompt.partial(vision="a man sitting in a room")
        search = TavilySearchResults()
        
        tools=[search]

        agent = create_tool_calling_agent(
            llm = model,
            prompt=prompt,
            tools=tools
        )

        agentExecutor = AgentExecutor(
            agent=agent,
            tools=tools
            # verbose=True
        )


        # print("Here")
        if user_input.lower() == 'exit':
            break
        print("test1")
        # response = process_chat(agentExecutor, user_input, chat_history)
        response = agentExecutor.invoke({"input": user_input, "chat_history": chat_history})
        print("test2")
        chat_history.append(HumanMessage(content=user_input))
        # print("test3")
        chat_history.append(AIMessage(content=response['output']))

        print("Assisstant: ", response)

if __name__ == "__main__":

    with concurrent.futures.ThreadPoolExecutor() as executor:
        future1 = executor.submit(main1)
        future2 = executor.submit(main2)
        future1.result()
        future2.result()
        # main2()
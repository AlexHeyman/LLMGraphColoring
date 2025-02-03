import os
import sys
import httpx
# from meta_ai_api import MetaAI
import fireworks.client
from openai import OpenAI
import anthropic
import google.generativeai as genai


if 'FIREWORKSAI_API_KEY' in os.environ:
  fireworks.client.api_key = os.environ['FIREWORKSAI_API_KEY']

if 'GOOGLE_API_KEY' in os.environ:
  genai.configure(api_key=os.environ['GOOGLE_API_KEY'])


class LanguageModel:
  
  def new_conversation(self):
    raise NotImplementedError


class Conversation:
  
  def send_and_receive(self, message, temperature=0):
    raise NotImplementedError


class DummyModel(LanguageModel):
  
  def __init__(self):
    pass
  
  def new_conversation(self):
    return DummyModelConversation()


class DummyModelConversation(Conversation):
  
  def __init__(self):
    pass
  
  def send_and_receive(self, message, temperature=0):
    return 'Dummy model response'


'''
class Llama3MetaAI(LanguageModel):
  
  def __init__(self):
    self.client = MetaAI()
  
  def new_conversation(self):
    return Llama3MetaAIConversation(self.client)


class Llama3MetaAIConversation(Conversation):
  
  def __init__(self, client):
    self.client = client
    self.has_sent = False
  
  def send_and_receive(self, message, temperature=0):
    # Note that the temperature parameter doesn't do anything here yet
    if self.has_sent:
      raise RuntimeError(
          'Llama3MetaAIConversation does not support multiple messages')
    self.has_sent = True
    return self.client.prompt(message=message)['message']
'''


class FireworksAIModel(LanguageModel):
  
  def __init__(self, model_name, max_tokens=4096):
    self.model_name = model_name
    self.max_tokens = max_tokens
  
  def new_conversation(self):
    return FireworksAIModelConversation(self)


class FireworksAIModelConversation(Conversation):
  
  def __init__(self, model):
    self.model = model
    self.has_sent = False
  
  def send_and_receive(self, message, temperature=0):
    if self.has_sent:
      raise RuntimeError(
          'FireworksAIModelConversation does not support multiple messages')
    self.has_sent = True
    
    response = fireworks.client.ChatCompletion.create(
      self.model.model_name,
      messages=[{'role': 'user', 'content': message}],
      temperature=temperature,
      n=1,
      max_tokens=self.model.max_tokens
    )
    return response.choices[0].message.content


class OpenAIModel(LanguageModel):
  
  def __init__(self, model_name, timeout=60):
    # timeout is in seconds
    self.client = OpenAI(timeout=httpx.Timeout(timeout))
    self.model_name = model_name
  
  def new_conversation(self):
    return OpenAIModelConversation(self)


class OpenAIModelConversation(Conversation):
  
  def __init__(self, model):
    self.model = model
    self.has_sent = False
  
  def send_and_receive(self, message, temperature=0):
    if self.has_sent:
      raise RuntimeError(
          'OpenAIModelConversation does not support multiple messages')
    self.has_sent = True
    
    try:
      response = self.model.client.chat.completions.create(
        model=self.model.model_name,
        temperature=temperature,
        messages=[{'role': 'user', 'content': message}]
      )
      return response.choices[0].message.content
    except httpx.TimeoutException as e:
      raise RuntimeError('Response timeout')


class AnthropicModel(LanguageModel):
  
  def __init__(self, model_name, max_tokens=4096):
    self.client = anthropic.Anthropic()
    self.model_name = model_name
    self.max_tokens = max_tokens
  
  def new_conversation(self):
    return AnthropicModelConversation(self)


class AnthropicModelConversation(Conversation):
  
  def __init__(self, model):
    self.model = model
    self.has_sent = False
  
  def send_and_receive(self, message, temperature=0):
    if self.has_sent:
      raise RuntimeError(
          'AnthropicModelConversation does not support multiple messages')
    self.has_sent = True
    
    try:
      response = self.model.client.messages.create(
        model=self.model.model_name,
        max_tokens=self.model.max_tokens,
        temperature=temperature,
        messages=[{'role': 'user',
                   'content': [{'type': 'text', 'text': message}]
                   }]
      )
      return response.content[0].text
    except anthropic.InternalServerError as e:
      if e.error.type == 'overloaded_error':
        raise RuntimeError('Server overloaded')
      else:
        raise RuntimeError(str(e))


class GoogleModel(LanguageModel):
  
  def __init__(self, model_name):
    self.client = genai.GenerativeModel(model_name)
    self.model_name = model_name
  
  def new_conversation(self):
    return GoogleModelConversation(self)


class GoogleModelConversation(Conversation):

  def __init__(self, model):
    self.model = model
    self.has_sent = False

  def send_and_receive(self, message, temperature=0):
    if self.has_sent:
      raise RuntimeError(
          'GoogleModelConversation does not support multiple messages')
    self.has_sent = True
    
    response = self.model.client.generate_content(message,
      generation_config=genai.GenerationConfig(temperature=temperature))
    
    return response.text

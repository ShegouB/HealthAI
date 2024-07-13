
from langflow.base.data.utils import IMG_FILE_TYPES, TEXT_FILE_TYPES
from langflow.base.io.chat import ChatComponent
from langflow.io import (
    DropdownInput,
    FileInput,
    MessageTextInput,
    MultilineInput,
    Output,
    PromptInput,
)
from langflow.schema.message import Message
from langflow.base.prompts.api_utils import process_prompt_template
from langflow.io import Output, PromptInput
from langflow.schema.message import Message
from langflow.template.utils import update_template_values

import asyncio
import streamlit as st

# ----------------- ChatInput Component ---------------- #


class ChatInput(ChatComponent):
    display_name = "Chat Input"
    description = "Get chat inputs from the Playground."
    icon = "ChatInput"
    name = "ChatInput"

    inputs = [
        MultilineInput(
            name="input_value",
            display_name="Text",
            value="",
            info="Message to be passed as input.",
        ),
        DropdownInput(
            name="sender",
            display_name="Sender Type",
            options=["Machine", "User"],
            value="User",
            info="Type of sender.",
            advanced=True,
        ),
        MessageTextInput(
            name="sender_name",
            display_name="Sender Name",
            info="Name of the sender.",
            value="User",
            advanced=True,
        ),
        MessageTextInput(
            name="session_id",
            display_name="Session ID",
            info="Session ID for the message.",
            advanced=True,
        ),
        FileInput(
            name="files",
            display_name="Files",
            file_types=TEXT_FILE_TYPES + IMG_FILE_TYPES,
            info="Files to be sent with the message.",
            advanced=True,
            is_list=True,
        ),
    ]
    outputs = [
        Output(display_name="Message", name="message", method="message_response"),
    ]

    def message_response(self) -> Message:
        message = Message(
            text=self.input_value,
            sender=self.sender,
            sender_name=self.sender_name,
            session_id=self.session_id,
            files=self.files,
        )
        if self.session_id and isinstance(
            message, Message
        ) and isinstance(message.text, str):
            self.store_message(message)
            # Assign the message to self.message.value
            self.message.value = message

        self.status = message
        return message


# ----------------- PromptComponent Component ---------------- #
from langflow.custom import Component

class PromptComponent(Component):
    display_name: str = "Prompt"
    description: str = "Create a prompt template with dynamic variables."
    icon = "prompts"
    trace_type = "prompt"
    name = "Prompt"

    inputs = [
        PromptInput(name="template", display_name="Template"),
    ]

    outputs = [
        Output(display_name="Prompt Message", name="prompt", method="build_prompt"),
    ]

    async def build_prompt(self) -> Message:
        prompt = await Message.from_template_and_variables(template=self.template, **self._attributes) 
        self.status = prompt.text
        return prompt

    def post_code_processing(self, new_build_config: dict, current_build_config: dict):
        """
        This function is called after the code validation is done.
        """
        frontend_node = super().post_code_processing(
            new_build_config, current_build_config
        )
        template = frontend_node["template"]["template"]["value"]
        _ = process_prompt_template(
            template=template,
            name="template",
            custom_fields=frontend_node["custom_fields"],
            frontend_node_template=frontend_node["template"],
        )
        # Now that template is updated, we need to grab any values that were set in the current_build_config
        # and update the frontend_node with those values
        update_template_values(
            new_template=frontend_node, previous_template=current_build_config["template"]
        )
        return frontend_node


# ----------------- GoogleGenerativeAIComponent Component ---------------- #
from pydantic.v1 import SecretStr

from langflow.base.constants import STREAM_INFO_TEXT
from langflow.base.models.model import LCModelComponent
from langflow.field_typing import LanguageModel
from langflow.inputs import (
    BoolInput,
    DropdownInput,
    FloatInput,
    IntInput,
    MessageInput,
    SecretStrInput,
    StrInput,
)


class GoogleGenerativeAIComponent(LCModelComponent):
    display_name = "Google Generative AI"
    description = "Generate text using Google Generative AI."
    icon = "GoogleGenerativeAI"
    name = "GoogleGenerativeAIModel"

    inputs = [
        MessageInput(name="input_value", display_name="Input"),
        IntInput(
            name="max_output_tokens",
            display_name="Max Output Tokens",
            info="The maximum number of tokens to generate.",
        ),
        DropdownInput(
            name="model",
            display_name="Model",
            info="The name of the model to use.",
            options=[
                "gemini-1.5-pro",
                "gemini-1.5-flash",
                "gemini-1.0-pro",
                "gemini-1.0-pro-vision",
            ],
            value="gemini-1.5-pro",
        ),
        SecretStrInput(
            name="google_api_key",
            display_name="Google API Key",
            info="The Google API Key to use for the Google Generative AI.",
        ),
        FloatInput(
            name="top_p",
            display_name="Top P",
            info="The maximum cumulative probability of tokens to consider when sampling.",
            advanced=True,
        ),
        FloatInput(name="temperature", display_name="Temperature", value=0.1),
        BoolInput(
            name="stream", display_name="Stream", info=STREAM_INFO_TEXT, advanced=True
        ),
        IntInput(
            name="n",
            display_name="N",
            info="Number of chat completions to generate for each prompt. Note that the API may not return the full n completions if duplicates are generated.",
            advanced=True,
        ),
        StrInput(
            name="system_message",
            display_name="System Message",
            info="System message to pass to the model.",
            advanced=True,
        ),
        IntInput(
            name="top_k",
            display_name="Top K",
            info="Decode using top-k sampling: consider the set of top_k most probable tokens. Must be positive.",
            advanced=True,
        ),
    ]

    def build_model(self) -> LanguageModel:  # type: ignore[type-var]
        try:
            from langchain_google_genai import ChatGoogleGenerativeAI
        except ImportError:
            raise ImportError(
                "The 'langchain_google_genai' package is required to use the Google Generative AI model."
            )

        google_api_key = self.google_api_key
        model = self.model
        max_output_tokens = self.max_output_tokens
        temperature = self.temperature
        top_k = self.top_k
        top_p = self.top_p
        n = self.n

        output = ChatGoogleGenerativeAI(  # type: ignore
            model=model,
            max_output_tokens=max_output_tokens or None,
            temperature=temperature,
            top_k=top_k or None,
            top_p=top_p or None,
            n=n or 1,
            google_api_key=SecretStr(google_api_key),
        )

        return output  # type: ignore


# ----------------- ChatOutput Component ---------------- #
from langflow.base.io.chat import ChatComponent
from langflow.io import DropdownInput, MessageTextInput, Output
from langflow.schema.message import Message


class ChatOutput(ChatComponent):
    display_name = "Chat Output"
    description = "Display a chat message in the Playground."
    icon = "ChatOutput"
    name = "ChatOutput"

    inputs = [
        MessageTextInput(
            name="input_value",
            display_name="Text",
            info="Message to be passed as output.",
        ),
        DropdownInput(
            name="sender",
            display_name="Sender Type",
            options=["Machine", "User"],
            value="Machine",
            advanced=True,
            info="Type of sender.",
        ),
        MessageTextInput(
            name="sender_name",
            display_name="Sender Name",
            info="Name of the sender.",
            value="AI",
            advanced=True,
        ),
        MessageTextInput(
            name="session_id",
            display_name="Session ID",
            info="Session ID for the message.",
            advanced=True,
        ),
        MessageTextInput(
            name="data_template",
            display_name="Data Template",
            value="{text}",
            advanced=True,
            info="Template to convert Data to Text. If left empty, it will be dynamically set to the Data's text key.",
        ),
    ]
    outputs = [
        Output(display_name="Message", name="message", method="message_response"),
    ]

    def message_response(self) -> Message:
        message = Message(
            text=self.input_value,
            sender=self.sender,
            sender_name=self.sender_name,
            session_id=self.session_id,
        )
        if self.session_id and isinstance(
            message, Message
        ) and isinstance(message.text, str):
            self.store_message(message)
            self.message.value = message

        self.status = message
        return message


# Assuming these variables are defined somewhere in your application
google_api_key = st.secrets["GOOGLE_API_KEY"]
prompt_template = "You are a virtual assistant specializing in healthcare. Your purpose is to provide accurate and reliable information about medical conditions, treatments, medications, and healthy lifestyle advice. You can also help users manage their medication schedules, provide reminders for doctor appointments, and offer emotional support. Always ensure your responses are clear, concise, and respectful. When you encounter a question you cannot answer, advise the user to consult with a healthcare professional"


# ---------------- Streamlit UI ----------------- #
async def main():  # Make main asynchronous
    st.title("HealthAI: Your Virtual Health Assistant")

    st.sidebar.title("Menu")
    menu_options = ["Home", "Chat with HealthAI", "About"]
    choice = st.sidebar.selectbox("Select an option", menu_options)

    if choice == "Home":
        st.write("Welcome to HealthAI! Use the menu to navigate.")
    elif choice == "Chat with HealthAI":
        st.subheader("Chat with HealthAI")
    

        # Initialize chat history
        if "messages" not in st.session_state:
            st.session_state.messages = []

        # Initialize components
        chat_input = ChatInput()
        prompt_component = PromptComponent()
        google_ai_component = GoogleGenerativeAIComponent(
            google_api_key=google_api_key
        )
        chat_output = ChatOutput()

        # Get user input
        user_input = st.text_input("You:")
        if user_input:
            # Add user message to chat history
            st.session_state.messages.append({"role": "user", "content": user_input})

            # Prepare input for the model
            chat_input.input_value = user_input
            message = chat_input.message_response()
            prompt_component.template = prompt_template
            prompt = await prompt_component.build_prompt()

            # Set the prompt as the system message
            google_ai_component.system_message = prompt.text 

            # Provide user input as a message
            google_ai_component.input_value = message

            # Execute the model and retrieve the response 
            try:
                response_message = await google_ai_component.atext_response()  
            except AttributeError:
                response_message = google_ai_component.text_response() 

            chat_output.input_value = response_message
            output_message = chat_output.message_response()

            st.session_state.messages.append({"role": "assistant", "content": output_message.text})
        
        # Display chat history
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

    elif choice == "About":
        st.write("HealthAI is your virtual assistant for health-related inquiries.")

if __name__ == "__main__":
    asyncio.run(main())  # Run the asynchronous main function
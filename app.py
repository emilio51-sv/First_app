
import sys
import os
import pandas as pd
import json
from crewai import Crew, Agent, Task, Process
import streamlit as st

# Load environment variables
openai_api_key = st.secrets["OPENAI_API_KEY"]
GPT4_LLM = ChatOpenAI(model_name="gpt-4o-mini", temperature=0.5)

# Streamlit UI setup
st.title('Blog Writer')
st.write("""
This app is designed to write a blog post automatically.
""")

# Input from user for the query
query = st.text_input("What is the topic?")

if st.button("Start Research"):
    if not query:
        st.error("Please enter a topic to proceed.")
    else:
        # Define agents
        Content_planner = Agent(
            role="Content Planner",
            goal=f"Plan engaging and factually accurate content on {query}",
            backstory=f"You're working on planning a blog article about the topic: {query}.",
            verbose=True,
            memory=True,
            llm="GPT4_LLM",
        )

        Content_writer = Agent(
            role="Content Writer",
            goal=f"Write insightful and factually accurate opinion piece about the topic: {query}",
            backstory=f"You're working on writing a new opinion piece about the topic: {query}.",
            verbose=True,
            memory=True,
            llm="GPT4_LLM",
        )

        editor = Agent(
            role="Editor",
            goal="Edit a given blog post to align with the writing style of the organization.",
            backstory="You are an editor who reviews blog posts.",
            verbose=True,
            memory=True,
            llm="GPT4_LLM",
        )

        # Define tasks
        plan = Task(
            description="Plan content for the given topic.",
            expected_output="A comprehensive content plan document with an outline.",
            agent=Content_planner,
        )

        write = Task(
            description="Write a blog post based on the content plan.",
            expected_output="A well-written blog post in markdown format.",
            agent=Content_writer,
        )

        edit = Task(
            description="Proofread the blog post.",
            expected_output="A polished blog post, ready for publication.",
            agent=editor,
        )

        # Form the crew
        crew = Crew(
            agents=[Content_planner, Content_writer, editor],
            tasks=[plan, write, edit],
            process=Process.sequential,
            full_output=True,
        )

        # Kickoff the crew process
        with st.spinner("Generating your blog post..."):
            results = crew.kickoff()

        # Display results
        st.subheader("Generated Blog Content")
        for step, result in enumerate(results, 1):
            st.write(f"### Step {step}: {result['task']}")
            st.write(result['output'])

        st.success("Blog post generation completed!")

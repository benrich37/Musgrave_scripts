# Musgrave Scripts

Note 1 - Add any ideas or suggestion to this README until the repo is in working order

Note 2 - this repo is currently public so we can use the CODEOWNERS feature. If we want to invest in github pro or something, we can change it to private while keeping CODEOWNERS functionality

### Getting Started

1. Create your own directory named after yourself to hold your scripts
2. Add a line in CODEOWNERS to protect your directory
    (ie to protect my script folder, I've added the line 'BenRich/* @benrich37', which tells GitHub that everything (wildcard \*) under the directory "BenRich/" is owned by the GitHub user @benrich37)
    By doing this, any changes you make to files in someone else's directory will have to be verified by the owner of that directory in order to be committed to the main branch.
3. ?

### Comments and Documentation
Maybe some loose rules on how we document and organize our scripts for easy readability? Like a preferred docustring format?

### Workflows / Unit Tests
These would be super good with shared scripts that receive a lot of edits from a lot of different users, but has the major downside that if we made this repo private we would have to pay out of pocket for any computational time it takes to run the test checks


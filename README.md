# TDT13 - NLP Project
## Sentiment analysis and Classifcation of Oslo stock exchange messages

### Data stats
* Total 460 166 messages and 22 message features
* All `id`s are equal to `messageId`s
* There are null values for columns `title`, `body`, `issuerSign`, `instrumentName` and `instrumentFullName`.
    * **Title**: 1 message is null
    * **Body**: 74545 messages are null
    * **issuerSign**: 458 messages (still have `issuer_id` so can potentially be inferred
    * **instrumentName**: 273 833 messages
    * **instrumentFullName**: 294 518 messages (~70%)
    
## Language identification
### Title classification
* 74 267 titles are not classified as either english norwegian as the most likely
    * 25 142 have neither English, Norwegian or Swedish as top 3 language predicted
    
### Body classification
* Some bodies use \r\n, and not just \n.
    
